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
    <div className="border-t border-gray-700 mt-4 pt-3 space-y-3">
      <div className="flex gap-2">
        <input
          value={newComment}
          onChange={(e) => setNewComment(e.target.value)}
          placeholder="Write a comment..."
          className="flex-1 bg-transparent border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-red-500"
        />
        <button
          onClick={handleAddComment}
          className="bg-red-600 hover:bg-red-700 text-white px-3 rounded-lg transition-all"
        >
          <Send className="h-4 w-4" />
        </button>
      </div>

      {loading ? (
        <p className="text-sm text-gray-400">Loading comments...</p>
      ) : comments.length === 0 ? (
        <p className="text-sm text-gray-400">No comments yet.</p>
      ) : (
        <div className="space-y-2">
          {comments.map((c) => (
            <div
              key={c._id}
              className="bg-gray-800/50 rounded-lg p-2 text-sm flex justify-between"
            >
              <span>
                <strong className="text-red-400">{c.user.username}:</strong>{" "}
                {c.text}
              </span>
              <span className="text-gray-500 text-xs">
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
