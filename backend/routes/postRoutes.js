const express = require("express");
const router = express.Router();
const Post = require("../models/Post");
const jwt = require("jsonwebtoken");

const authMiddleware = (req, res, next) => {
  const authHeader = req.header("Authorization");
  if (!authHeader?.startsWith("Bearer ")) {
    return res.status(401).json({ message: "No token, authorization denied" });
  }

  try {
    const token = authHeader.split(" ")[1];
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded;
    next();
  } catch (err) {
    res.status(401).json({ message: "Invalid or expired token" });
  }
};

router.post("/posts", authMiddleware, async (req, res) => {
  try {
    const { title, description, category, location, image } = req.body;

    if (!title || !description || !category || !location) {
      return res.status(400).json({ message: "All fields are required" });
    }

    console.log("New Post Received:", req.body);

    // NLP model placeholder
    const isIssue = true;

    const newPost = new Post({
      title,
      description,
      category,
      location,
      image,
      isIssue,
      user: req.user.id,
    });

    await newPost.save();

    console.log(" Post saved successfully:", newPost._id);
    res.status(201).json({ message: "Post created successfully", post: newPost });
  } catch (err) {
    console.error(" Post creation error:", err);
    res.status(500).json({ message: "Server error", error: err.message });
  }
});

router.post("/posts", authMiddleware, async (req, res) => {
  try {
    const { title, description, category, location, image } = req.body;

    if (!title || !description || !category || !location) {
      return res.status(400).json({ message: "All fields are required" });
    }

    console.log("New Post Received:", req.body);

    const newPost = new Post({
      title,
      description,
      category,
      location,
      image,
      user: req.user.id,
    });

    await newPost.save();
    await newPost.populate("user", "username email profilePic");
    console.log("Post saved successfully:", newPost._id);
    res.status(201).json({ message: "Post created successfully", post: newPost });
  } catch (err) {
    console.error("Post creation error:", err);
    res.status(500).json({ message: "Server error", error: err.message });
  }
});

router.get("/myposts", authMiddleware, async (req, res) => {
  try {
    const posts = await Post.find({ user: req.user.id })
      .populate("user", "username email profilePic") 
      .sort({ createdAt: -1 });
    res.status(200).json(posts);
  } catch (err) {
    console.error("Error fetching user posts:", err);
    res.status(500).json({ message: "Failed to fetch your posts" });
  }
});


router.get("/posts/mine", authMiddleware, async (req, res) => {
  try {
    // req.user.id comes from your JWT decoded token
    const posts = await Post.find({ user: req.user.id }).sort({ createdAt: -1 });
    res.status(200).json(posts);
  } catch (err) {
    console.error("Error fetching user's posts:", err);
    res.status(500).json({ message: "Failed to fetch user posts" });
  }
});


module.exports = router;
