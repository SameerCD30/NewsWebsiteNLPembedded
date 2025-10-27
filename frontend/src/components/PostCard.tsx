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
import { Popover, PopoverContent, PopoverTrigger } from "./ui/popover";
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
import { Separator } from "./ui/separator";
import { CommentSection } from "./CommentSection";

interface PostCardProps {
  author: string;
  timestamp: string;
  content: string;
  upvotes: number;
  comments: number;
  tag?: string;
  image?: string;
  location?: string;
}

export const PostCard = ({
  author,
  timestamp,
  content,
  upvotes,
  comments,
  tag,
  image,
  location,
}: PostCardProps) => {
  const [votes, setVotes] = useState(upvotes);
  const [showVoteBreakdown, setShowVoteBreakdown] = useState(false);
  const [isReported, setIsReported] = useState(false);
  const [reportDialog, setReportDialog] = useState(false);
  const [showComments, setShowComments] = useState(false);

  const getInitials = (name: string) => {
    return name
      .split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase();
  };

  const handleUpvote = () => setVotes((prev) => prev + 1);
  const handleReport = () => {
    setIsReported(true);
    setReportDialog(false);
  };

  const toggleComments = () => setShowComments((prev) => !prev);

  const voteBreakdown = {
    local: Math.floor(votes * 0.5),
    state: Math.floor(votes * 0.3),
    national: Math.floor(votes * 0.2),
  };

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
          <Avatar className="h-11 w-11 ring-2 ring-primary/20 ring-offset-2 ring-offset-background transition-all duration-200 group-hover:ring-primary/40">
            <AvatarFallback className="bg-gradient-to-br from-primary/30 to-primary/10 text-primary font-bold">
              {getInitials(author)}
            </AvatarFallback>
          </Avatar>
          <div className="flex flex-col">
            <span className="font-semibold text-foreground text-base">{author}</span>
            <span className="text-sm text-muted-foreground/80">{timestamp}</span>
          </div>
        </div>

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
      </div>

      {/* Content */}
      <p className="text-foreground/90 mb-5 leading-relaxed text-[15px]">{content}</p>

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
      {location && <p className="text-sm text-muted-foreground mb-4">📍 {location}</p>}

      <Separator className="mb-3" />

      {/* Footer */}
      <div className="flex items-center gap-6 pt-2">
{/* Simple Upvote Button */}
<button
  aria-label="Upvote post"
  onClick={handleUpvote}
  className="flex items-center gap-2 text-muted-foreground hover:text-primary transition-all duration-200 hover:scale-110"
>
  <ArrowUp className="h-5 w-5 transition-transform duration-200" />
  <span className="text-sm font-semibold">{votes}</span>
</button>


        {/* Comments */}
        <button
          aria-label="View comments"
          onClick={toggleComments}
          className="flex items-center gap-2 text-muted-foreground hover:text-primary transition-all duration-200 hover:scale-110"
        >
          <MessageCircle className="h-5 w-5" />
          <span className="text-sm font-semibold">{comments}</span>
        </button>

        {/* Dropdown */}
        <DropdownMenu>
  <DropdownMenuTrigger asChild>
    <button
      aria-label="More actions"
      className="flex items-center gap-2 ml-auto text-muted-foreground 
                 hover:text-primary transition-all duration-300 
                 hover:scale-110 active:scale-95 p-2 rounded-full 
                 hover:bg-muted/30 backdrop-blur-sm"
    >
      <MoreVertical className="h-5 w-5" />
    </button>
  </DropdownMenuTrigger>

  <DropdownMenuContent
    align="end"
    className="mt-2 w-44 rounded-xl border border-border/50 bg-card/90 
               shadow-lg backdrop-blur-md text-sm p-1 
               animate-in fade-in-0 zoom-in-95"
  >
    {/* Share */}
    <Dialog>
      <DialogTrigger asChild>
        <DropdownMenuItem
          className="cursor-pointer flex items-center gap-2 px-3 py-2.5 rounded-lg 
                     text-foreground/90 transition-all duration-300 
                     hover:bg-blue-500/15 hover:text-blue-600"
        >
          <Share2 className="h-4 w-4 text-blue-500" />
          <span>Share</span>
        </DropdownMenuItem>
      </DialogTrigger>

      {/* Share Modal */}
     <DialogContent
  className="bg-card/95 backdrop-blur-xl border border-border/50 
             shadow-2xl rounded-2xl w-[380px] p-6 
             animate-in fade-in-0 zoom-in-95 
             data-[state=open]:animate-in 
             data-[state=closed]:animate-out 
             data-[state=open]:zoom-in-95 
             data-[state=closed]:zoom-out-95 
             duration-300"
>
  <DialogHeader>
    <DialogTitle className="text-lg font-semibold text-foreground text-center">
      Share Post
    </DialogTitle>
  </DialogHeader>

  <p className="text-sm text-muted-foreground mb-5 text-center">
    Choose a platform to share or copy the post link:
  </p>

  <div className="flex flex-col gap-3">
    <Button
      variant="outline"
      onClick={() => {
        navigator.clipboard.writeText(window.location.href);
        alert("✅ Post link copied to clipboard!");
      }}
      className="justify-start text-sm hover:bg-blue-500/15 hover:text-blue-600 
                 transition-all duration-200 shadow-sm hover:shadow-md"
    >
      🔗 Copy Link
    </Button>

    <Button
      variant="outline"
      onClick={() =>
        window.open(
          `https://twitter.com/intent/tweet?url=${encodeURIComponent(window.location.href)}`,
          "_blank"
        )
      }
      className="justify-start text-sm hover:bg-blue-400/20 hover:text-blue-500 
                 transition-all duration-200 shadow-sm hover:shadow-md"
    >
      🐦 Share on X (Twitter)
    </Button>

    <Button
      variant="outline"
      onClick={() =>
        window.open(
          `https://api.whatsapp.com/send?text=${encodeURIComponent(window.location.href)}`,
          "_blank"
        )
      }
      className="justify-start text-sm hover:bg-green-500/15 hover:text-green-600 
                 transition-all duration-200 shadow-sm hover:shadow-md"
    >
      💬 Share on WhatsApp
    </Button>

    <Button
      variant="outline"
      onClick={() =>
        (window.location.href = `mailto:?subject=Check this post&body=${encodeURIComponent(window.location.href)}`)
      }
      className="justify-start text-sm hover:bg-orange-400/15 hover:text-orange-500 
                 transition-all duration-200 shadow-sm hover:shadow-md"
    >
      📧 Share via Email
    </Button>
  </div>
</DialogContent>

    </Dialog>

    {/* Report */}
    <Dialog open={reportDialog} onOpenChange={setReportDialog}>
      <DialogTrigger asChild>
        <DropdownMenuItem
          className="cursor-pointer flex items-center gap-2 px-3 py-2.5 rounded-lg 
                     text-danger transition-all duration-300 
                     hover:bg-red-500/15 hover:text-red-600"
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
          Are you sure you want to report this post? It will be reviewed by moderators or concerned authorities.
        </p>

        <DialogFooter className="mt-4 flex justify-end gap-2">
          <Button variant="outline" onClick={() => setReportDialog(false)} className="border-border/60">
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

        {isReported && (
          <span className="flex items-center gap-1.5 text-sm text-red-600 ml-2">
            <CheckCircle2 className="h-4 w-4" /> Reported
          </span>
        )}
      </div>

      {/* Comment Section */}
      {showComments && (
        <div className="mt-3">
          <CommentSection />
        </div>
      )}
    </article>
  );
};
