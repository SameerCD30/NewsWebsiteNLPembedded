const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
require("dotenv").config();

const authRoutes = require("./routes/authRoutes");
const postRoutes = require("./routes/postRoutes");

const app = express();

app.use(express.json({ limit: "100mb" }));
app.use(express.urlencoded({ limit: "100mb", extended: true }));

app.use(
  cors({
    origin: ["http://localhost:8080", "http://localhost:5173"],
    methods: ["GET", "POST"],
    credentials: true,
  })
);

mongoose
  .connect(process.env.MONGO_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  })
  .then(() => console.log("MongoDB Connected"))
  .catch((err) => console.error("MongoDB Connection Error:", err));

app.use("/api/auth", authRoutes);
app.use("/api", postRoutes);

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

const PORT = process.env.PORT || 8081;
app.listen(PORT, () =>
  console.log(`Server running on http://localhost:${PORT}`)
);
