require("dotenv").config();
const express = require("express");
const multer = require("multer");
const neo4j = require("neo4j-driver");
const path = require("path");
const fs = require("fs");
const pdfParse = require("pdf-parse");
const app = express();
const port = process.env.PORT || 5000;
const cors = require('cors');
const nlp = require('compromise');
const axios = require('axios'); // Install axios: npm install axios

app.use(cors());
app.use(express.json());

// Neo4j connection
const driver = neo4j.driver(process.env.NEO4J_URI);
const uploadsDir = "./uploads";

if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir);

const storage = multer.diskStorage({
    destination: uploadsDir,
    filename: (req, file, cb) => cb(null, file.originalname),
});

const upload = multer({
    storage,
    limits: { fileSize: 10 * 1024 * 1024 },
    fileFilter: (req, file, cb) => {
        if (path.extname(file.originalname).toLowerCase() === ".pdf") {
            cb(null, true);
        } else {
            cb(new Error("Only PDF files are allowed!"), false);
        }
    },
});

app.post("/api/upload", upload.single("file"), async (req, res) => {
    if (!req.file) return res.status(400).json({ message: "No file uploaded." });

    const filepath = req.file.path;
    const filename = req.file.filename;
    let session;

    try {
        session = driver.session();
        const extractedText = await extractPDF(filepath);

        if (!extractedText) throw new Error("Failed to extract text from PDF.");

        await saveDocumentToNeo4j(session, filename, extractedText);

        res.json({ message: "PDF uploaded and processed successfully!", filename });
    } catch (error) {
        console.error("Error processing file:", error);
        res.status(500).json({ message: "File processing error", error: error.message });
    } finally {
        if (session) await session.close();
        fs.unlink(filepath, (err) => {
            if (err) console.error("Error deleting file:", err);
        });
    }
});

async function extractPDF(filepath) {
    try {
        const data = await pdfParse(fs.readFileSync(filepath));
        return data.text.trim();
    } catch (error) {
        console.error("PDF extraction error:", error);
        return "";
    }
}

async function saveDocumentToNeo4j(session, filename, textContent) {
    try {
        const query = `
                CREATE (d:Document { name: $filename, text: $textContent, uploadedAt: datetime() })
                RETURN id(d) AS documentId
            `;
        const result = await session.run(query, { filename, textContent });
        console.log("Document saved to Neo4j:", result.records[0].get("documentId").toNumber());
        return result.records[0].get("documentId").toNumber();
    } catch (error) {
        console.error("Error saving document to Neo4j:", error);
        throw error;
    }
}

app.listen(port, () => console.log(`Server running on port ${port}`));


app.post('/api/chat', async (req, res) => {
    try {
        const question = req.body.question;

        if (!question || typeof question !== 'string' || question.trim() === '') {
            return res.status(400).json({ reply: 'Invalid question.' });
        }

        const doc = nlp(question);
        console.log("Compromise Doc:", doc);
        const terms = doc.terms().json();

        console.log("Compromise Terms:", terms);

        let movieTitleTerms = [];
        let attributeTerm = null;
        const stopWords = ['what', 'are', 'the', 'in', 'movie', 'of', 'a', 'an', 'who','is','when','was','which']; // Added 'who'
        const attributeList = ['director', 'actor', 'genre', 'plot', 'released', 'year', 'rating', 'poster', 'directors', 'actors', 'genres', 'acted', 'release year', 'released year', 'release date', 'released date', 'date','languages','country'];
        // Identify movie title terms and the attribute term
        terms.forEach((term) => {
            const text = term.text.toLowerCase();
            if (attributeList.includes(text)) {
                if (text === 'directors') {
                    attributeTerm = 'director';
                } else if (text === 'actors') {
                    attributeTerm = 'actor';
                } else if (text === 'genres') {
                    attributeTerm = 'genre';
                } else if (text === 'acted') {
                    attributeTerm = 'actor'; // Treat "acted" as "actor"
                } else {
                    attributeTerm = text;
                }
            } else if (!stopWords.includes(text)) {
                movieTitleTerms.push(text);
            }
        });

        if (!movieTitleTerms.length || !attributeTerm) {
            return res.json({ reply: "I couldn't identify the movie or attribute you're asking about." });
        }

        let neo4jQuery = 'MATCH (m:Movie)';
        let whereClause = [];
        let params = {};

        const movieTitle = movieTitleTerms.join(' ');
        whereClause.push('toLower(m.title) = $title');
        params.title = movieTitle;

        if (whereClause.length > 0) {
            neo4jQuery += ' WHERE ' + whereClause.join(' AND ');
        }

        let returnAttribute = `m.${attributeTerm}`;
        if (attributeTerm === "actor") {
            returnAttribute = "m.main_actor";
        } else if (attributeTerm === "rating") {
            returnAttribute = "m.imdbRating";
        } else if (attributeTerm === "release year"||attributeTerm === "released year") {
            returnAttribute = "m.year"; // Use 'm.year'
        } else if (attributeTerm === "date" || attributeTerm === "release date"|| attributeTerm === "released date"|| attributeTerm === "year release") {
            returnAttribute = "m.released"; // Use 'm.released'
        } else if (attributeTerm === "released" || attributeTerm === "year" ) {
            returnAttribute = `m.${attributeTerm}`;
        }
        neo4jQuery += ` RETURN ${returnAttribute}`;

        const session = driver.session();
        const neo4jResult = await session.run(neo4jQuery, params);
        console.log("Neo4j Result:", neo4jResult.records);
        session.close();

        let response = '';
        neo4jResult.records.forEach((record) => {
            response += `- ${attributeTerm.charAt(0).toUpperCase() + attributeTerm.slice(1)}: ${record.get(returnAttribute)}\n`;
        });

        res.json({ reply: response || "No results found." });

    } catch (error) {
        console.error('Error in /api/chat:', error);
        res.status(500).json({ reply: 'An error occurred.' });
    }
});


process.on("SIGINT", async () => {
    console.log("Shutting down gracefully...");
    try {
        if (driver) {
            await driver.close();
            console.log("Neo4j driver closed.");
        }
    } catch (error) {
        console.error("Error closing Neo4j driver:", error);
    }
    console.log("Exiting process.");
    process.exit(0);
});