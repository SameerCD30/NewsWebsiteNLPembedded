router.get("/posts", async (req, res) => {
  try {
    const posts = await Post.find()
        .populate("user", "username email") 
        .sort({ createdAt: -1 });

    res.status(200).json(posts);
  } catch (err) {
    console.error("Error fetching posts:", err);
    res.status(500).json({ message: "Server error", error: err.message });
  }
});
