import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";
import {
  ArrowUp,
  MessageCircle,
  MoreVertical,
  Share2,
  Flag,
  CheckCircle2,
  MapPin,
} from "lucide-react";
import { Avatar, AvatarFallback } from "./ui/avatar";
import { Badge } from "./ui/badge";
import { Separator } from "./ui/separator";
import { CommentSection } from "./CommentSection";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "./ui/dropdown-menu";
import { Button } from "./ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogTrigger,
} from "./ui/dialog";
import { upvotePost, removeUpvote } from "@/api/api";
import { Post } from "@/types/post"; 

interface PostCardProps {
  post: Post;
}

const REPORT_THRESHOLD = 3;

export const PostCard = ({ post }: PostCardProps) => {
  const [votes, setVotes] = useState(post.upvotes || 0);
  const [isUpvoted, setIsUpvoted] = useState(post.isUpvoted || false);
  const [isReported, setIsReported] = useState(false);
  const [reportDialog, setReportDialog] = useState(false);
  const [showComments, setShowComments] = useState(false);
  const [reportCount, setReportCount] = useState(0);

  const author = post.user?.username || "Anonymous";
  const timestamp = new Date(post.createdAt).toLocaleString("en-IN", {
    dateStyle: "medium",
    timeStyle: "short",
  });

  const handleUpvote = async () => {
    const token = localStorage.getItem("token");
    if (!token) {
      alert("Please log in to upvote.");
      return;
    }
    try {
      if (isUpvoted) {
        const res = await removeUpvote(post._id);
        if (res.status === 200) {
          setVotes((prev) => Math.max(prev - 1, 0));
          setIsUpvoted(false);
        }
      } else {
        const res = await upvotePost(post._id);
        if (res.status === 200) {
          setVotes((prev) => prev + 1);
          setIsUpvoted(true);
        }
      }
    } catch (err: any) {
      if (err.response?.status === 401) {
        alert("Please log in to upvote.");
      } else {
        alert(err.response?.data?.message || "Something went wrong.");
      }
    }
  };

  const handleReport = async () => {
    if (isReported) return;
    setIsReported(true);
    setReportDialog(false);
    try {
      const res = await fetch(`/api/report/${post._id}`, { method: "POST" });
      if (!res.ok) throw new Error("Failed to report");
      const data = await res.json();
      const newCount =
        typeof data.reportCount === "number"
          ? data.reportCount
          : reportCount + 1;
      setReportCount(newCount);
    } catch {
      setReportCount((prev) => prev + 1);
    }
  };

  const getInitials = (name: string) =>
    name
      .split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase();

  let shortLocation = "";

  if (typeof post.location === "string") {
    shortLocation = post.location.split(",").slice(0, 3).join(",");
  } else if (typeof post.location === "object" && post.location !== null) {
    const { city, state, country } = post.location as any;
    shortLocation = [city, state, country].filter(Boolean).join(", ");
  }


  const tagColors: Record<string, string> = {
    Municipal: "bg-blue-100 text-blue-700 border-blue-300",
    Water: "bg-cyan-100 text-cyan-700 border-cyan-300",
    Electricity: "bg-yellow-100 text-yellow-700 border-yellow-300",
    Police: "bg-purple-100 text-purple-700 border-purple-300",
    Other: "bg-gray-100 text-gray-700 border-gray-300",
  };

  return (
    <article className="bg-card rounded-xl border border-border/60 p-6 hover:border-primary/50 hover:shadow-xl transition-all duration-300 group">
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <Avatar className="h-11 w-11 ring-2 ring-primary/20 ring-offset-2 ring-offset-background">
            <AvatarFallback className="bg-gradient-to-br from-primary/30 to-primary/10 text-primary font-bold">
              {getInitials(author)}
            </AvatarFallback>
          </Avatar>
          <div className="flex flex-col">
            <span className="font-semibold text-foreground text-base">
              {author}
            </span>
            <span className="text-sm text-muted-foreground/80">{timestamp}</span>
            <AnimatePresence>
              {shortLocation && (
                <motion.span
                  initial={{ opacity: 0, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                  transition={{ duration: 0.3 }}
                  className="text-sm text-muted-foreground/70 flex items-center gap-1 mt-0.5"
                >
                  <MapPin className="h-4 w-4 text-orange-500" />
                  {shortLocation}
                </motion.span>
              )}
            </AnimatePresence>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {post.category && (
            <Badge
              variant="outline"
              className={`${
                tagColors[post.category] ||
                "bg-primary/5 text-primary border-primary/40"
              } px-3 py-1 font-medium`}
            >
              {post.category}
            </Badge>
          )}
          {reportCount >= REPORT_THRESHOLD && (
            <Badge className="bg-yellow-100 text-yellow-800 border-yellow-200 font-medium">
              Potentially Fake
            </Badge>
          )}
        </div>
      </div>

      <h2 className="text-lg font-semibold text-foreground mb-3">
        {post.title}
      </h2>

      <p className="text-foreground/90 mb-5 leading-relaxed text-[15px]">
        {post.description}
      </p>

      {post.image && (
        <div className="mb-5 -mx-6 px-6">
          <img
            src={post.image}
            alt="Post image"
            className="w-full h-80 object-cover rounded-xl shadow-lg hover:scale-[1.02] hover:shadow-2xl transition-all duration-300"
            loading="lazy"
          />
        </div>
      )}

      <Separator className="mb-3" />

      <div className="flex items-center gap-6 pt-2">
        <motion.button
          aria-label={isUpvoted ? "Remove upvote" : "Upvote post"}
          onClick={handleUpvote}
          whileTap={{ scale: 1.2 }}
          className="relative flex items-center gap-2 transition-all duration-200 hover:scale-110"
        >
          <motion.div
            initial={false}
            animate={{
              scale: isUpvoted ? [1, 1.3, 1] : [1.1, 1, 1],
              rotate: isUpvoted ? [0, -15, 0] : 0,
            }}
            transition={{ duration: 0.3 }}
            className="relative flex items-center justify-center"
          >
            <ArrowUp
              className={`h-6 w-6 transition-colors duration-300 ${
                isUpvoted
                  ? "text-orange-500"
                  : "text-gray-400 group-hover:text-orange-400"
              }`}
            />
            <AnimatePresence>
              {isUpvoted && (
                <motion.div
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ scale: 1.2, opacity: 1 }}
                  exit={{ scale: 0, opacity: 0 }}
                  transition={{ duration: 0.25 }}
                  className="absolute inset-0 bg-orange-500 rounded-full blur-md opacity-40"
                />
              )}
            </AnimatePresence>
          </motion.div>
          <motion.span
            animate={{ color: isUpvoted ? "#f97316" : "#9ca3af" }}
            transition={{ duration: 0.2 }}
            className="text-sm font-semibold"
          >
            {votes}
          </motion.span>
        </motion.button>

        <button
          aria-label="View comments"
          onClick={() => setShowComments((prev) => !prev)}
          className="flex items-center gap-2 text-muted-foreground hover:text-primary transition-all duration-200 hover:scale-110"
        >
          <MessageCircle className="h-5 w-5" />
          <span className="text-sm font-semibold">
            {post.comments?.length || 0}
          </span>
        </button>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              aria-label="More actions"
              className="flex items-center gap-2 ml-auto text-muted-foreground hover:text-primary transition-all duration-300 hover:scale-110 p-2 rounded-full hover:bg-muted/30 backdrop-blur-sm"
            >
              <MoreVertical className="h-5 w-5" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            align="end"
            className="mt-2 w-44 rounded-xl border border-border/50 bg-card/90 shadow-lg backdrop-blur-md text-sm p-1 animate-in fade-in-0 zoom-in-95"
          >
            <DropdownMenuItem
              onClick={() => {
                navigator.clipboard.writeText(window.location.href);
                alert("Post link copied to clipboard!");
              }}
              className="cursor-pointer flex items-center gap-2 px-3 py-2.5 rounded-lg hover:bg-blue-500/15 hover:text-blue-600"
            >
              <Share2 className="h-4 w-4 text-blue-500" />
              <span>Share</span>
            </DropdownMenuItem>
            <Dialog open={reportDialog} onOpenChange={setReportDialog}>
              <DialogTrigger asChild>
                <DropdownMenuItem
                  className="cursor-pointer flex items-center gap-2 px-3 py-2.5 rounded-lg text-danger hover:bg-red-500/15 hover:text-red-600"
                  onSelect={(e) => e.preventDefault()}
                >
                  <Flag className="h-4 w-4 text-red-500" />
                  <span>{isReported ? "Reported" : "Report"}</span>
                </DropdownMenuItem>
              </DialogTrigger>
              <DialogContent className="bg-card/95 backdrop-blur-xl border border-border/50 shadow-xl rounded-2xl">
                <DialogHeader>
                  <DialogTitle className="text-lg font-semibold text-foreground">
                    Report Post
                  </DialogTitle>
                </DialogHeader>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Are you sure you want to report this post? It will be reviewed
                  by moderators or concerned authorities.
                </p>
                <DialogFooter className="mt-4 flex justify-end gap-2">
                  <Button
                    variant="outline"
                    onClick={() => setReportDialog(false)}
                    className="border-border/60"
                  >
                    Cancel
                  </Button>
                  <Button
                    variant="destructive"
                    onClick={handleReport}
                    className="bg-red-600 hover:bg-red-700 transition-all shadow-md hover:shadow-lg"
                  >
                    Confirm Report
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </DropdownMenuContent>
        </DropdownMenu>
        {isReported && (
          <span className="flex items-center gap-1.5 text-sm text-red-600 ml-2 font-medium">
            <CheckCircle2 className="h-4 w-4" /> Reported
          </span>
        )}
      </div>

      {showComments && (
        <div className="mt-3">
          <CommentSection postId={post._id} />
        </div>
      )}
    </article>
  );
};

export default PostCard;
