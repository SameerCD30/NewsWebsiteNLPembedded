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
    if (!token) return alert("Please log in to upvote.");

    try {
      if (isUpvoted) {
        const res = await removeUpvote(post._id);
        if (res.status === 200) {
          setVotes((v) => v - 1);
          setIsUpvoted(false);
        }
      } else {
        const res = await upvotePost(post._id);
        if (res.status === 200) {
          setVotes((v) => v + 1);
          setIsUpvoted(true);
        }
      }
    } catch (err) {
      alert("Something went wrong.");
    }
  };

  const handleReport = async () => {
    if (isReported) return;
    setIsReported(true);
    setReportDialog(false);
    try {
      const res = await fetch(`/api/report/${post._id}`, { method: "POST" });
      if (!res.ok) throw new Error("Fail");
      const data = await res.json();
      setReportCount(data.reportCount || reportCount + 1);
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

  // LOCATION FORMAT
  let shortLocation = "";
  if (typeof post.location === "string") {
    shortLocation = post.location.split(",").slice(0, 3).join(", ");
  } else if (typeof post.location === "object" && post.location !== null) {
    const { city, state, country } = post.location as any;
    shortLocation = [city, state, country].filter(Boolean).join(", ");
  }

  const tagColors: Record<string, string> = {
    Municipal: "bg-blue-600/20 text-blue-300 border-blue-500/40",
    Water: "bg-cyan-600/20 text-cyan-300 border-cyan-500/40",
    Electricity: "bg-yellow-600/20 text-yellow-300 border-yellow-500/40",
    Police: "bg-purple-600/20 text-purple-300 border-purple-500/40",
    Other: "bg-gray-600/20 text-gray-300 border-gray-500/40",
  };

  return (
    <article
      className="bg-[#0d1117]/80 backdrop-blur-xl p-6 rounded-2xl 
      border border-blue-700/20 shadow-[0_0_15px_rgba(0,102,255,0.1)]
      hover:border-blue-600/40 hover:shadow-[0_0_25px_rgba(0,102,255,0.2)]
      transition-all duration-300"
    >
      
      {/* HEADER */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <Avatar className="h-11 w-11 ring-2 ring-blue-500/30 ring-offset-2 ring-offset-[#0b0f16]">
            <AvatarFallback className="bg-blue-600/20 text-blue-300 font-bold">
              {getInitials(author)}
            </AvatarFallback>
          </Avatar>

          <div className="flex flex-col">
            <span className="font-semibold text-blue-300 text-base">
              {author}
            </span>

            <span className="text-sm text-gray-400 flex items-center gap-2 mt-0.5">
              {timestamp}

              {shortLocation && (
                <>
                  <span className="text-gray-600">|</span>
                  <span className="flex items-center gap-1 text-gray-400">
                    <MapPin className="h-4 w-4 text-blue-500" />
                    {shortLocation}
                  </span>
                </>
              )}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {post.category && (
            <Badge
              variant="outline"
              className={`${tagColors[post.category]} px-3 py-1 font-medium`}
            >
              {post.category}
            </Badge>
          )}

          {reportCount >= REPORT_THRESHOLD && (
            <Badge className="bg-yellow-500/20 text-yellow-300 border-yellow-500/40 font-medium">
              Potentially Fake
            </Badge>
          )}
        </div>
      </div>

      {/* TITLE */}
      <h2 className="text-lg font-semibold text-blue-200 mb-3">
        {post.title}
      </h2>

      {/* DESCRIPTION */}
      <p className="text-gray-300 mb-5 leading-relaxed text-[15px]">
        {post.description}
      </p>

      {/* IMAGE */}
      {post.image && (
        <div className="mb-5 -mx-6 px-6">
          <img
            src={post.image}
            className="w-full h-80 object-cover rounded-xl shadow-lg 
            hover:scale-[1.02] hover:shadow-[0_0_20px_rgba(0,102,255,0.3)]
            transition-all duration-300"
            loading="lazy"
          />
        </div>
      )}

      <Separator className="mb-3 bg-blue-700/20" />

      {/* ACTIONS */}
      <div className="flex items-center gap-6 pt-2">

        {/* UPVOTE */}
        <motion.button
          onClick={handleUpvote}
          whileTap={{ scale: 1.2 }}
          className="relative flex items-center gap-2 hover:scale-110 transition-all"
        >
          <motion.div
            initial={false}
            animate={{
              scale: isUpvoted ? [1, 1.3, 1] : 1,
              rotate: isUpvoted ? [0, -15, 0] : 0,
            }}
            transition={{ duration: 0.3 }}
            className="relative"
          >
            <ArrowUp
              className={`h-6 w-6 ${
                isUpvoted ? "text-blue-400" : "text-gray-400"
              }`}
            />

            <AnimatePresence>
              {isUpvoted && (
                <motion.div
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ scale: 1.2, opacity: 1 }}
                  exit={{ scale: 0, opacity: 0 }}
                  transition={{ duration: 0.25 }}
                  className="absolute inset-0 bg-blue-500 rounded-full blur-md opacity-40"
                />
              )}
            </AnimatePresence>
          </motion.div>

          <motion.span
            animate={{ color: isUpvoted ? "#60a5fa" : "#9ca3af" }}
            transition={{ duration: 0.2 }}
            className="text-sm font-semibold"
          >
            {votes}
          </motion.span>
        </motion.button>

        {/* COMMENTS */}
        <button
          onClick={() => setShowComments((x) => !x)}
          className="flex items-center gap-2 text-gray-400 hover:text-blue-400 hover:scale-110 transition-all"
        >
          <MessageCircle className="h-5 w-5" />
          <span className="text-sm font-semibold">
            {post.comments?.length || 0}
          </span>
        </button>

        {/* MORE MENU */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button className="ml-auto p-2 rounded-full text-gray-400 
            hover:text-blue-400 hover:bg-blue-900/20 hover:scale-110 transition">
              <MoreVertical className="h-5 w-5" />
            </button>
          </DropdownMenuTrigger>

          <DropdownMenuContent
            align="end"
            className="mt-2 w-44 bg-[#0b0f16]/90 backdrop-blur-xl 
            border border-blue-700/40 rounded-xl shadow-xl p-1"
          >
            <DropdownMenuItem
              onClick={() => {
                navigator.clipboard.writeText(window.location.href);
                alert("Post link copied!");
              }}
              className="cursor-pointer flex items-center gap-2 px-3 py-2.5 
              rounded-lg hover:bg-blue-600/15 hover:text-blue-400 transition"
            >
              <Share2 className="h-4 w-4 text-blue-400" />
              Share
            </DropdownMenuItem>

            <Dialog open={reportDialog} onOpenChange={setReportDialog}>
              <DialogTrigger asChild>
                <DropdownMenuItem
                  className="cursor-pointer flex items-center gap-2 px-3 py-2.5 
                  rounded-lg text-red-300 hover:bg-red-600/15 hover:text-red-400"
                  onSelect={(e) => e.preventDefault()}
                >
                  <Flag className="h-4 w-4 text-red-400" />
                  {isReported ? "Reported" : "Report"}
                </DropdownMenuItem>
              </DialogTrigger>

              <DialogContent className="bg-[#0b0f16]/95 backdrop-blur-2xl border border-blue-700/40 shadow-xl rounded-2xl">
                <DialogHeader>
                  <DialogTitle className="text-blue-300">
                    Report Post
                  </DialogTitle>
                </DialogHeader>

                <p className="text-sm text-gray-400">
                  Are you sure you want to report this post?
                </p>

                <DialogFooter className="mt-4 flex justify-end gap-2">
                  <Button variant="outline" onClick={() => setReportDialog(false)}>
                    Cancel
                  </Button>

                  <Button
                    variant="destructive"
                    onClick={handleReport}
                    className="bg-red-600 hover:bg-red-700"
                  >
                    Confirm Report
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </DropdownMenuContent>
        </DropdownMenu>

        {isReported && (
          <span className="flex items-center gap-1.5 text-sm text-red-400 ml-2 font-medium">
            <CheckCircle2 className="h-4 w-4" /> Reported
          </span>
        )}
      </div>

      {/* COMMENTS */}
      {showComments && (
        <div className="mt-3">
          <CommentSection postId={post._id} />
        </div>
      )}
    </article>
  );
};

export default PostCard;
