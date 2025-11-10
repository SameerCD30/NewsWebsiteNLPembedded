const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
require("dotenv").config();

const authRoutes = require("./routes/authRoutes");

const app = express();
app.use(
  cors({
    origin: ["http://localhost:8080", "http://localhost:5173"], // allow both ports
    methods: ["GET", "POST"],
    credentials: true,
  })
);

app.use(express.json());

mongoose
  .connect(process.env.MONGO_URI)
  .then(() => console.log("MongoDB Connected"))
  .catch((err) => console.error("MongoDB Connection Error:", err));

app.use("/api/auth", authRoutes);

app.get("/", (req, res) => {
  res.send("Root Page");
});

app.get("/health", async (req, res) => {
  const dbState = mongoose.connection.readyState;
  res.json({
    status: "Backend running",
    database: dbState === 1 ? "MongoDB connected" : "MongoDB not connected",
    timestamp: new Date(),
  });
});

const PORT = 8081;
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
