// require('dotenv').config(); // If using .env file

// const neo4j = require('neo4j-driver');
// console.log("Neo4j URI:", process.env.NEO4J_URI);
// console.log("Neo4j Username:", process.env.NEO4J_USERNAME);
// console.log("Neo4j Password (masked):", process.env.NEO4J_PASSWORD ? "*****" : "Not Set"); // Mask the password for security

// const driver = neo4j.driver(
//     process.env.NEO4J_URI, // No fallback here; force environment variables
//     neo4j.auth.basic(process.env.NEO4J_USERNAME, process.env.NEO4J_PASSWORD)
// );

// driver.verifyConnectivity()
// .then(() => {
//     console.log('Successfully connected to Neo4j');
//     driver.close();
// })
// .catch(error => {
//     console.error('Error connecting to Neo4j:', error);
//     driver.close();
// });

// require('dotenv').config(); // If using .env for environment variables

// const neo4j = require('neo4j-driver');

// console.log("Neo4j URI:", process.env.NEO4J_URI);
// console.log("Neo4j Username:", process.env.NEO4J_USERNAME);
// console.log("Neo4j Password (masked):", process.env.NEO4J_PASSWORD ? "*****" : "Not Set");

// const driver = neo4j.driver(
//     process.env.NEO4J_URI || "neo4j://localhost:7687", // Consistent URI!
//     neo4j.auth.basic(process.env.NEO4J_USERNAME, process.env.NEO4J_PASSWORD)
// );

// driver.verifyConnectivity()
// .then(() => {
//     console.log('Successfully connected to Neo4j');
//     driver.close();
// })
// .catch(error => {
//     console.error('Error connecting to Neo4j:', error);
//     console.error('Error Code:', error.code);
//     console.error('Error Message:', error.message);
//     driver.close();
// });


// const neo4j = require('neo4j-driver');

// // Hardcoded credentials for TESTING ONLY - REMOVE FOR PRODUCTION!
// const driver = neo4j.driver(
//     "neo4j://localhost:7687", // Or neo4j://127.0.0.1:7687 if needed
//     neo4j.auth.basic("neo4j", "admin@1234") // Replace with your actual password!
// );

// driver.verifyConnectivity()
// .then(() => {
//     console.log('Successfully connected to Neo4j');
//     driver.close();
// })
// .catch(error => {
//     console.error('Error connecting to Neo4j:', error);
//     driver.close();
// });

// const neo4j = require('neo4j-driver');

// const driver = neo4j.driver(
//     "neo4j://localhost:7687" // URI only, no authentication
// );

// async function testConnection() {
//     try {
//         const session = driver.session();
//         const result = await session.run('RETURN 1');
//         console.log('Successfully connected to Neo4j (without authentication)!');
//         console.log('Query Result:', result.records[0].get(0));
//         await session.close();
//         await driver.close();
//     } catch (error) {
//         console.error('Error connecting to Neo4j:', error);
//         await driver.close();
//     }
// }

// testConnection();

const nlp = require('compromise');

const doc = nlp('John Smith is a person.');
const entities = doc.ents().json();

console.log(entities);