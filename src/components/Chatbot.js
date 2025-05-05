import React, { useState, useEffect } from "react";
import 'react-chat-widget/lib/styles.css';
import { Widget, addResponseMessage } from "react-chat-widget";

const Chatbot = () => {
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    addResponseMessage("Hello! How can I assist you today?");
  }, []);

  const handleNewUserMessage = (newMessage) => {
    setMessages([...messages, { user: true, text: newMessage }]);
    addResponseMessage("You said: " + newMessage);
  };

  return (
    <div>
      <Widget
        handleNewUserMessage={handleNewUserMessage}
        title="Chatbot"
        subtitle="I'm here to help!"
      />
    </div>
  );
};

export default Chatbot;