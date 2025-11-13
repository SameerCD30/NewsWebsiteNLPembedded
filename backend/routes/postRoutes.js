const express = require("express");
const router = express.Router();
const Post = require("../models/Post");
const jwt = require("jsonwebtoken");
const statesData = require("../utils/stateCities");
const normalizeLocation = require("../utils/normalizeLocation");

const axios = require("axios");


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
  } catch {
    res.status(401).json({ message: "Invalid or expired token" });
  }
};

// CREATE POST
router.post("/posts", authMiddleware, async (req, res) => {
  try {
    const { title, description, category, location, image } = req.body;

    if (!title || !description || !category || !location) {
      return res.status(400).json({ message: "All fields are required" });
    }

    try {
      const ml = await axios.post("http://127.0.0.1:5000/predict", {
        text: description
      });

      if (!ml.data.is_issue) {
        return res.status(400).json({
          message:
            "This description does not appear to be a valid public issue.",
          confidence: ml.data.confidence,
        });
      }
    } catch (err) {
      console.error("ML Server Error:", err.message);
      return res.status(500).json({
        message:
          "Issue verification service unavailable. Please try again later.",
      });
    }

    let formattedLocation = {
      city: "",
      state: "",
      country: "India",
      address: typeof location === "string" ? location : location.address || "",
    };

    if (typeof location === "string") {
      const parts = location.split(",").map((p) => p.trim());
      formattedLocation.city = parts[0] || "Unknown City";
      formattedLocation.state = parts[1] || "Unknown State";
    } else if (typeof location === "object") {
      formattedLocation.city = location.city || "Unknown City";
      formattedLocation.state = location.state || "Unknown State";
      formattedLocation.country = location.country || "India";
    }

    const { cityKey, stateKey } = normalizeLocation(
      formattedLocation.city,
      formattedLocation.state
    );
    formattedLocation.cityKey = cityKey;
    formattedLocation.stateKey = stateKey;

    // Save post
    const newPost = new Post({
      title,
      description,
      category,
      location: formattedLocation,
      image,
      user: req.user.id,
    });

    await newPost.save();
    await newPost.populate("user", "username email profilePic");

    res
      .status(201)
      .json({ message: "Post created successfully", post: newPost });
  } catch (err) {
    res.status(500).json({ message: "Server error", error: err.message });
  }
});


// FETCH POSTS (local/state/national)
router.get("/posts", async (req, res) => {
  try {
    const { scope, city, state } = req.query;
    let query = {};

    if (scope === "local" && city && state) {
      const { cityKey, stateKey } = normalizeLocation(city, state);

      const stateData = statesData[stateKey];
      const cityData = stateData ? stateData[cityKey] : null;
      const pinList = cityData?.pincodes || [];

      query.$or = [
        { "location.city": { $regex: new RegExp(city, "i") } },
        { "location.state": { $regex: new RegExp(state, "i") } },
        ...(pinList.length > 0
          ? [{ "location.address": { $regex: new RegExp(pinList.join("|"), "i") } }]
          : []),
      ];
    }else if (scope === "state" && state) {
      const { stateKey } = normalizeLocation("", state);
      const stateData = statesData[stateKey];

      if (stateData) {
        const allCities = Object.values(stateData).map((c) => c.city.toLowerCase());
        query.$or = [
          { "location.stateKey": stateKey },
          { "location.cityKey": { $in: allCities } },
        ];
      } else {
        query["location.state"] = { $regex: new RegExp(state, "i") };
      }
    }else if (scope === "national") {
      query = {};
    }

    let posts = await Post.find(query)
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
      } catch {
        userId = null;
      }
    }

    posts = posts.map((p) => {
      const upvotedBy = Array.isArray(p.upvotedBy) ? p.upvotedBy : [];
      const isUpvoted = userId
        ? upvotedBy.some((uid) => String(uid) === String(userId))
        : false;
      return { ...p, isUpvoted };
    });

    res.status(200).json(posts);
  } catch (err) {
    res.status(500).json({ message: "Failed to load posts" });
  }
});

// MY POSTS
router.get("/myposts", authMiddleware, async (req, res) => {
  try {
    const posts = await Post.find({ user: req.user.id })
      .populate("user", "username email profilePic")
      .sort({ createdAt: -1 });
    res.status(200).json(posts);
  } catch {
    res.status(500).json({ message: "Failed to fetch your posts" });
  }
});

// UPVOTE
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
      return res.status(400).json({ message: "Already upvoted" });
    }
    res.status(200).json({ message: "Upvoted", upvotes: updated.upvotes });
  } catch {
    res.status(500).json({ message: "Server error" });
  }
});

// REMOVE UPVOTE
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
      return res.status(400).json({ message: "You haven’t upvoted this post yet" });
    }
    res.status(200).json({ message: "Upvote removed", upvotes: updated.upvotes });
  } catch {
    res.status(500).json({ message: "Server error" });
  }
});

module.exports = router;
