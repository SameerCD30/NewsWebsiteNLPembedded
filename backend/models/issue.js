const mongoose = require("mongoose");

const issueSchema = new mongoose.Schema({
  title: { type: String, required: true },
  description: { type: String, required: true },
  category: { type: String, enum: ["local", "state", "national"], required: true },
  location: {
    city: String,
    state: String,
    country: String
  },
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User" },
  createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model("Issue", issueSchema);
