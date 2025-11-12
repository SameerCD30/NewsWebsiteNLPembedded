const mongoose = require("mongoose");

const commentSchema = new mongoose.Schema({
  user: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  text: { type: String, required: true },
  createdAt: { type: Date, default: Date.now },
});

const locationSchema = new mongoose.Schema({
  address: { type: String },
  city: { type: String, required: true },
  state: { type: String, required: true },
  country: { type: String, default: "India" },
  pincode: { type: String, default: "000000" },
  lat: { type: Number },
  lng: { type: Number },
  cityKey: { type: String, lowercase: true, trim: true },
  stateKey: { type: String, lowercase: true, trim: true },
});

const postSchema = new mongoose.Schema({
  title: { type: String, required: true },
  description: { type: String, required: true },
  category: {
    type: String,
    enum: ["Municipal", "Water", "Electricity", "Police", "Other"],
    required: true,
  },
  location: { type: locationSchema, required: true },
  image: { type: String },
  user: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  upvotes: { type: Number, default: 0 },
  upvotedBy: [{ type: mongoose.Schema.Types.ObjectId, ref: "User" }],
  comments: [commentSchema],
  createdAt: { type: Date, default: Date.now },
});

postSchema.index({ "location.stateKey": 1 });
postSchema.index({ "location.cityKey": 1 });

module.exports = mongoose.model("Post", postSchema);
