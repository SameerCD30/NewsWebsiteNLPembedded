import { useState } from "react";
import {
  ArrowUp,
  MessageCircle,
  MoreVertical,
  Share2,
  Flag,
  CheckCircle2,
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

interface Post {
  _id: string;
  title: string;
  description: string;
  category: string;
  image?: string;
  location?: string;
  createdAt: string;
  user?: {
    username?: string;
    email?: string;
  };
  upvotes?: number;
  comments?: any[];
}

interface PostCardProps {
  post: Post;
}

// Number of reports required to show the "Potentially fake" label
const REPORT_THRESHOLD = 3;

export const PostCard = ({ post }: PostCardProps) => {
  const [votes, setVotes] = useState(post.upvotes || 0);
  const [isReported, setIsReported] = useState(false);
  const [reportDialog, setReportDialog] = useState(false);
  const [showComments, setShowComments] = useState(false);
  const [reportCount, setReportCount] = useState(0);

  const author = post.user?.username || "Anonymous";
  const timestamp = new Date(post.createdAt).toLocaleString("en-IN", {
    dateStyle: "medium",
    timeStyle: "short",
  });
  const content = post.description;
  const tag = post.category;
  const image = post.image;
  const location = post.location;

  const getInitials = (name: string) =>
    name
      .split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase();

  const handleUpvote = () => setVotes((prev) => prev + 1);

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
    } catch (err) {
      console.error(err);
      setReportCount((prev) => prev + 1);
    }
  };

  const toggleComments = () => setShowComments((prev) => !prev);

  const tagColors: Record<string, string> = {
    Fake: "bg-red-100 text-red-700 border-red-300",
    Real: "bg-green-100 text-green-700 border-green-300",
    Pending: "bg-yellow-100 text-yellow-700 border-yellow-300",
  };

  return (
    <article className="bg-card rounded-xl border border-border/60 p-6 hover:border-primary/50 hover:shadow-xl transition-all duration-300 group">
      {/* Header */}
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
          </div>
        </div>

        {/* Tag section */}
        <div className="flex items-center gap-2">
          {tag && (
            <Badge
              variant="outline"
              className={`${
                tagColors[tag] || "bg-primary/5 text-primary border-primary/40"
              } px-3 py-1 font-medium`}
            >
              {tag}
            </Badge>
          )}

          {reportCount >= REPORT_THRESHOLD && (
            <Badge className="bg-yellow-100 text-yellow-800 border-yellow-200 font-medium">
              Potentially Fake
            </Badge>
          )}
        </div>
      </div>

      {/* Content */}
      <p className="text-foreground/90 mb-5 leading-relaxed text-[15px]">
        {content}
      </p>

      {/* Image */}
      {image && (
        <div className="mb-5 -mx-6 px-6">
          <img
            src={image}
            alt="Post image"
            className="w-full h-80 object-cover rounded-xl shadow-lg hover:scale-[1.02] hover:shadow-2xl transition-all duration-300"
            loading="lazy"
          />
        </div>
      )}

      {/* Location */}
      {location && (
        <p className="text-sm text-muted-foreground mb-4">📍 {location}</p>
      )}

      <Separator className="mb-3" />

      {/* Footer */}
      <div className="flex items-center gap-6 pt-2">
        {/* Upvote */}
        <button
          aria-label="Upvote post"
          onClick={handleUpvote}
          className="flex items-center gap-2 text-muted-foreground hover:text-primary transition-all duration-200 hover:scale-110"
        >
          <ArrowUp className="h-5 w-5" />
          <span className="text-sm font-semibold">{votes}</span>
        </button>

        {/* Comments */}
        <button
          aria-label="View comments"
          onClick={toggleComments}
          className="flex items-center gap-2 text-muted-foreground hover:text-primary transition-all duration-200 hover:scale-110"
        >
          <MessageCircle className="h-5 w-5" />
          <span className="text-sm font-semibold">
            {post.comments?.length || 0}
          </span>
        </button>

        {/* Dropdown */}
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
            {/* Share */}
            <DropdownMenuItem
              onClick={() => {
                navigator.clipboard.writeText(window.location.href);
                alert(" Post link copied to clipboard!");
              }}
              className="cursor-pointer flex items-center gap-2 px-3 py-2.5 rounded-lg hover:bg-blue-500/15 hover:text-blue-600"
            >
              <Share2 className="h-4 w-4 text-blue-500" />
              <span>Share</span>
            </DropdownMenuItem>

            {/* Report */}
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

        {/* Reported Indicator */}
        {isReported && (
          <span className="flex items-center gap-1.5 text-sm text-red-600 ml-2 font-medium">
            <CheckCircle2 className="h-4 w-4" /> Reported
          </span>
        )}
      </div>


    </article>
  );
};
