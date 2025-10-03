const express = require("express"); 
const mongoose = require("mongoose");
const cors = require("cors");
require("dotenv").config();

const app = express();
app.use(cors());
app.use(express.json()); 

mongoose.connect(process.env.MONGO_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true
})
.then(() => console.log("MongoDB Connected"))
.catch(err => console.error("Error:", err));

app.get("/", (req, res) => {
    res.send("root Page");
});
app.get("/health", async (req, res) => {
  const dbState = mongoose.connection.readyState; // 0 = disconnected, 1 = connected
  res.json({
    status: "Backend running",
    database: dbState === 1 ? "MongoDB connected" : "MongoDB not connected",
    timestamp: new Date()
  });
});


const PORT = 8080;
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
