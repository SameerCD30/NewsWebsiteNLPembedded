const express = require("express");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const User = require("../models/User");
const authMiddleware = require("../middleware/authMiddleware");

const router = express.Router();

/**
 * ======================
 *  SIGNUP
 * ======================
 */
router.post("/signup", async (req, res) => {
  try {
    console.log("Signup request body:", req.body);

    const { username, email, password } = req.body;

    if (!username || !email || !password) {
      return res.status(400).json({ message: "All fields are required." });
    }

    // Check existing email
    const existingEmail = await User.findOne({ email });
    if (existingEmail) {
      return res.status(400).json({ message: "Email already registered." });
    }

    // Check existing username
    const existingUsername = await User.findOne({ username });
    if (existingUsername) {
      return res.status(400).json({ message: "Username already taken." });
    }

    // Hash password only ONCE
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    const newUser = new User({
      username,
      email,
      password: hashedPassword,
    });

    await newUser.save();

    return res.status(201).json({ message: "Signup successful." });
  } catch (err) {
    console.error("Signup Error:", err);
    return res
      .status(500)
      .json({ message: "Server error", error: err.message });
  }
});

/**
 * ======================
 *  LOGIN
 * ======================
 */
router.post("/login", async (req, res) => {
  try {
    console.log("Login body:", req.body);

    const { email, password } = req.body;

    if (!email || !password)
      return res.status(400).json({ message: "Email & password required." });

    // Allow login with either email OR username
    const user = await User.findOne({
      $or: [{ email }, { username: email }],
    });

    if (!user)
      return res.status(400).json({ message: "Invalid credentials." });

    // Compare plain password with hashed
    const isMatch = await bcrypt.compare(password, user.password);
    console.log("Password match result:", isMatch);

    if (!isMatch)
      return res.status(400).json({ message: "Invalid credentials." });

    // Generate JWT
    const token = jwt.sign(
      { id: user._id, email: user.email },
      process.env.JWT_SECRET,
      { expiresIn: "7d" }
    );

    return res.json({
      message: "Login successful.",
      token,
      user: {
        _id: user._id,
        username: user.username,
        email: user.email,
      },
    });
  } catch (err) {
    console.error("Login Error:", err);
    return res.status(500).json({ message: "Server error" });
  }
});

/**
 * ======================
 *  GET CURRENT USER
 * ======================
 */
router.get("/me", authMiddleware, async (req, res) => {
  try {
    const user = await User.findById(req.user.id).select("-password");

    if (!user)
      return res.status(404).json({ message: "User not found" });

    return res.json(user);
  } catch (err) {
    console.error("Fetch user error:", err);
    return res.status(500).json({ message: "Failed to fetch user" });
  }
});

module.exports = router;
