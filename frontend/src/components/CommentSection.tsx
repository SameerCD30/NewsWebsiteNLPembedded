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
    <div className="mt-4 border-t border-border/50 pt-4">
      <h3 className="font-semibold text-sm mb-3 text-muted-foreground">Comments</h3>

      <div className="space-y-3 max-h-64 overflow-y-auto pr-2">
        {comments.map((comment) => (
          <div
            key={comment.id}
            className="p-3 rounded-lg bg-muted/30 hover:bg-muted/40 transition-all"
          >
            <div className="flex justify-between items-center mb-1">
              <span className="font-medium text-sm text-primary/90">{comment.author}</span>
              <button
                onClick={() => handleLike(comment.id)}
                className="flex items-center gap-1 text-muted-foreground hover:text-primary transition"
              >
                <ThumbsUp className="h-4 w-4" />
                <span className="text-xs">{comment.likes}</span>
              </button>
            </div>
            <p className="text-sm text-foreground/90">{comment.content}</p>
          </div>
        ))}
      </div>

      <div className="flex items-center gap-2 mt-4">
        <Input
          placeholder="Write a comment..."
          value={newComment}
          onChange={(e) => setNewComment(e.target.value)}
          className="flex-1 text-sm"
        />
        <Button
          onClick={handleAddComment}
          size="sm"
          className="bg-primary hover:bg-primary/90"
        >
          <Send className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
};
