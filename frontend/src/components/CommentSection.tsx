import { useEffect, useState } from "react";
import { addComment, fetchComments } from "@/api/api";
import { Send } from "lucide-react";

interface Comment {
  _id: string;
  text: string;
  user: { username: string };
  createdAt: string;
}

export const CommentSection = ({ postId }: { postId: string }) => {
  const [comments, setComments] = useState<Comment[]>([]);
  const [newComment, setNewComment] = useState("");
  const [loading, setLoading] = useState(true);

  const loadComments = async () => {
    try {
      const res = await fetchComments(postId);
      setComments(res.data);
    } catch (err) {
      console.error("Error fetching comments:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleAddComment = async () => {
    const token = localStorage.getItem("token");
    if (!token) return alert("Please log in to comment.");
    if (!newComment.trim()) return;

    try {
      const res = await addComment(postId, newComment);
      setComments(res.data.comments);
      setNewComment("");
    } catch (err) {
      console.error("Error adding comment:", err);
    }
  };

  useEffect(() => {
    loadComments();
  }, []);

  return (
    <div className="border-t border-blue-700/30 mt-4 pt-4 space-y-4">

      {/* Comment Input */}
      <div className="flex gap-2">
        <input
          value={newComment}
          onChange={(e) => setNewComment(e.target.value)}
          placeholder="Write a comment..."
          className="flex-1 bg-[#0d1117] border border-blue-600/40 
          rounded-xl px-3 py-2 text-sm text-blue-200
          focus:outline-none focus:ring-2 focus:ring-blue-500/70
          placeholder-blue-400/50 transition"
        />

        <button
          onClick={handleAddComment}
          className="bg-blue-600 hover:bg-blue-700 
          text-white px-4 rounded-xl flex items-center justify-center
          shadow-[0_0_12px_rgba(0,102,255,0.5)]
          hover:shadow-[0_0_16px_rgba(0,102,255,0.7)] 
          transition-all"
        >
          <Send className="h-4 w-4" />
        </button>
      </div>

      {/* Comments */}
      {loading ? (
        <p className="text-sm text-blue-400">Loading comments...</p>
      ) : comments.length === 0 ? (
        <p className="text-sm text-blue-400/70">No comments yet.</p>
      ) : (
        <div className="space-y-3">
          {comments.map((c) => (
            <div
              key={c._id}
              className="bg-[#0f141c] border border-blue-700/30 
              rounded-xl p-3 text-sm flex justify-between 
              shadow-[0_0_10px_rgba(0,102,255,0.15)]"
            >
              <span className="text-blue-200">
                <strong className="text-blue-400">{c.user.username}:</strong>{" "}
                {c.text}
              </span>

              <span className="text-blue-400/60 text-xs ml-4 min-w-[110px] text-right">
                {new Date(c.createdAt).toLocaleString("en-IN", {
                  dateStyle: "short",
                  timeStyle: "short",
                })}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
