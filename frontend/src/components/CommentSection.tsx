import { useState } from "react";
import { Send, ThumbsUp } from "lucide-react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";

interface Comment {
  id: string;
  author: string;
  content: string;
  likes: number;
}

export const CommentSection = () => {
  const [comments, setComments] = useState<Comment[]>([
    { id: "1", author: "Amit Sharma", content: "I’ve seen this issue too!", likes: 2 },
    { id: "2", author: "Riya Verma", content: "Authorities should act fast.", likes: 1 },
  ]);
  const [newComment, setNewComment] = useState("");

  const handleAddComment = () => {
    if (!newComment.trim()) return;
    const newCommentObj = {
      id: Date.now().toString(),
      author: "You",
      content: newComment,
      likes: 0,
    };
    setComments([...comments, newCommentObj]);
    setNewComment("");
  };

  const handleLike = (id: string) => {
    setComments(
      comments.map((c) => (c.id === id ? { ...c, likes: c.likes + 1 } : c))
    );
  };

  return (
    <div className="mt-5 border-t border-red-500/30 pt-5 bg-gradient-to-b from-[#1a1a1a]/60 to-[#0e0e0e]/80 rounded-2xl p-4 backdrop-blur-md shadow-inner">
      <h3 className="font-semibold text-base mb-3 text-red-400 tracking-wide">
        Comments
      </h3>

      <div className="space-y-3 max-h-64 overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-red-600 scrollbar-track-transparent">
        {comments.map((comment) => (
          <div
            key={comment.id}
            className="p-3 rounded-xl bg-[#222]/60 border border-red-500/10 hover:border-red-500/40 transition-all duration-300 shadow-sm hover:shadow-md"
          >
            <div className="flex justify-between items-center mb-1">
              <span className="font-medium text-sm text-red-400/90">{comment.author}</span>
              <button
                onClick={() => handleLike(comment.id)}
                className="flex items-center gap-1 text-muted-foreground hover:text-red-500 transition-all"
              >
                <ThumbsUp className="h-4 w-4" />
                <span className="text-xs">{comment.likes}</span>
              </button>
            </div>
            <p className="text-sm text-gray-200">{comment.content}</p>
          </div>
        ))}
      </div>

      <div className="flex items-center gap-2 mt-4">
        <Input
          placeholder="Write a comment..."
          value={newComment}
          onChange={(e) => setNewComment(e.target.value)}
          className="flex-1 text-sm bg-[#111]/60 border border-red-500/30 text-gray-100 placeholder-gray-400 focus:border-red-500 focus:ring-0"
        />
        <Button
          onClick={handleAddComment}
          size="sm"
          className="bg-red-600 hover:bg-red-700 shadow-md hover:shadow-lg transition-all duration-200"
        >
          <Send className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
};
