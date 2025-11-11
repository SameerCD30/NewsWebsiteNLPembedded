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

    //Replace with real NLP model inference
    const isIssue = true; 

    if (!isIssue) {
      return res.status(400).json({ message: "Post rejected by verification" });
    }
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

router.get("/posts", async (req, res) => {
  try {
    let posts = await Post.find()
      .populate("user", "username email profilePic")
      .sort({ createdAt: -1 })
      .lean(); 

    const authHeader = req.header("Authorization");
    let userId = null;
    if (authHeader?.startsWith("Bearer ")) {
      try {
        const token = authHeader.split(" ")[1];
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        userId = decoded?.id || decoded?._id || null;
      } catch (err) {
        userId = null;
      }
    }

    if (userId) {
      posts = posts.map((p) => {
        const upvotedBy = Array.isArray(p.upvotedBy) ? p.upvotedBy : [];
        const isUpvoted = upvotedBy.some((uid) => String(uid) === String(userId));
        return { ...p, isUpvoted };
      });
    } else {
      posts = posts.map((p) => ({ ...p, isUpvoted: false }));
    }

    res.status(200).json(posts);
  } catch (err) {
    console.error("Error fetching posts:", err);
    res.status(500).json({ message: "Failed to load posts" });
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
    const posts = await Post.find({ user: req.user.id }).sort({ createdAt: -1 });
    res.status(200).json(posts);
  } catch (err) {
    console.error("Error fetching user's posts:", err);
    res.status(500).json({ message: "Failed to fetch user posts" });
  }
});

router.post("/posts/:id/upvote", authMiddleware, async (req, res) => {
  try {
    const userId = req.user.id;
    const updated = await Post.findOneAndUpdate(
      { _id: req.params.id, upvotedBy: { $ne: userId } },
      { $addToSet: { upvotedBy: userId }, $inc: { upvotes: 1 } },
      { new: true } 
    );

    if (!updated) {
      const exists = await Post.exists({ _id: req.params.id });
      if (!exists) return res.status(404).json({ message: "Post not found" });
      return res.status(400).json({ message: "You have already upvoted this post" });
    }

    res.status(200).json({ message: "Post upvoted", upvotes: updated.upvotes });
  } catch (err) {
    console.error("Error upvoting post:", err);
    res.status(500).json({ message: "Server error" });
  }
});

router.post("/posts/:id/unupvote", authMiddleware, async (req, res) => {
  try {
    const userId = req.user.id;
    const updated = await Post.findOneAndUpdate(
      { _id: req.params.id, upvotedBy: userId },
      { $pull: { upvotedBy: userId }, $inc: { upvotes: -1 } },
      { new: true }
    );

    if (!updated) {
      const exists = await Post.exists({ _id: req.params.id });
      if (!exists) return res.status(404).json({ message: "Post not found" });
      return res.status(400).json({ message: "You haven't upvoted this post yet" });
    }

    res.status(200).json({ message: "Upvote removed", upvotes: updated.upvotes });
  } catch (err) {
    console.error("Error removing upvote:", err);
    res.status(500).json({ message: "Server error" });
  }
});

router.post("/posts/:id/comments", authMiddleware, async (req, res) => {
  try {
    const { text } = req.body;
    if (!text) return res.status(400).json({ message: "Comment text is required" });

    const post = await Post.findById(req.params.id);
    if (!post) return res.status(404).json({ message: "Post not found" });

    const comment = {
      user: req.user.id,
      text,
    };

    post.comments.push(comment);
    await post.save();
    await post.populate("comments.user", "username email profilePic");

    res.status(201).json({ message: "Comment added", comments: post.comments });
  } catch (err) {
    console.error("Error adding comment:", err);
    res.status(500).json({ message: "Server error" });
  }
});

router.get("/posts/:id/comments", async (req, res) => {
  try {
    const post = await Post.findById(req.params.id)
      .populate("comments.user", "username email profilePic");
    if (!post) return res.status(404).json({ message: "Post not found" });
    res.status(200).json(post.comments);
  } catch (err) {
    console.error("Error fetching comments:", err);
    res.status(500).json({ message: "Server error" });
  }
});


module.exports = router;
