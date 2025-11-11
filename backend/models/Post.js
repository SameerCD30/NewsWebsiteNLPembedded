const mongoose = require("mongoose");

const postSchema = new mongoose.Schema({
  title: { type: String, required: true },
  description: { type: String, required: true },
  location: { type: String, required: true },
  category: {
    type: String,
    enum: ["Municipal", "Water", "Electricity", "Police", "Other"],
    required: true,
  },
  image: { type: String },
  
  user: { 
    type: mongoose.Schema.Types.ObjectId, 
    ref: "User", 
    required: true 
  },

  createdAt: { type: Date, default: Date.now },
});

module.exports = mongoose.model("Post", postSchema);
