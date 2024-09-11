import express from "express";
import multer from "multer";
import fs from "fs/promises";
import { Configuration, OpenAIApi } from "openai";
import * as dotenv from "dotenv";
dotenv.config();

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});

const openai = new OpenAIApi(configuration);

// Initialize Express server
const app = express();
app.use(express.json());

// Serve static files (frontend)
app.use(express.static("public"));

// Multer setup for file uploads
const upload = multer({ dest: "uploads/" });

// Tokenize and embed the text from the uploaded file
const embedText = async (text) => {
  const response = await openai.createEmbedding({
    model: "text-embedding-ada-002",
    input: text,
  });
  return response.data.data[0].embedding;
};

// Store the embedded data
const documents = [];

app.get("/", (req, res) => {
  res.sendFile(__dirname + "/public/index.html");
});

// Endpoint to get the list of uploaded files
app.get("/files", (req, res) => {
  const fileList = documents.map((doc) => doc.filename);
  res.json({ files: fileList });
});

// Endpoint to upload a file (company policies in text form)
app.post("/upload", upload.single("file"), async (req, res) => {
  try {
    const fileContent = await fs.readFile(req.file.path, "utf-8");
    const embedding = await embedText(fileContent);

    // Store the document and its embedding
    documents.push({
      filename: req.file.originalname,
      content: fileContent,
      embedding,
    });

    res.json({
      message: `File: ${req.file.originalname} uploaded and embedded successfully`,
    });
  } catch (error) {
    res.status(500).json({ error: "Failed to upload or process file" });
  }
});

// Endpoint for asking questions
app.post("/ask", async (req, res) => {
  const { question } = req.body;

  try {
    // Generate an embedding for the user's question
    const questionEmbedding = await embedText(question);

    // Find the most relevant document based on cosine similarity
    let bestMatch = null;
    let highestSimilarity = -Infinity;

    documents.forEach((doc) => {
      const similarity = cosineSimilarity(doc.embedding, questionEmbedding);
      if (similarity > highestSimilarity) {
        highestSimilarity = similarity;
        bestMatch = doc;
      }
    });

    // Ask the question to GPT-3.5 based on the best-matching document
    const answerResponse = await openai.createChatCompletion({
      model: "gpt-3.5-turbo",
      messages: [
        { role: "system", content: `You are a helpful assistant.` },
        { role: "user", content: `Here is some context: ${bestMatch.content}` },
        { role: "user", content: `Answer this question: ${question}` },
      ],
    });

    res.json({ answer: answerResponse.data.choices[0].message.content });
  } catch (error) {
    res.status(500).json({ error: "Error answering question" });
  }
});

// Cosine similarity calculation (for matching the best document)
const cosineSimilarity = (vecA, vecB) => {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (magnitudeA * magnitudeB);
};

// Start the server
app.listen(3000, () => {
  console.log("Server is running on port 3000");
});
