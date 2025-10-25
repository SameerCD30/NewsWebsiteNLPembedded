import { useState } from "react";
import {
  ArrowUp,
  MessageCircle,
  MoreVertical,
  Share2,
  Flag,
  CheckCircle2,
  Building2,
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
import { Label } from "./ui/label";
import { RadioGroup, RadioGroupItem } from "./ui/radio-group";
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
  const [authorityDialog, setAuthorityDialog] = useState(false);
  const [authorityTag, setAuthorityTag] = useState<string | null>(null);
  const [showComments, setShowComments] = useState(false); // 👈 NEW STATE

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

  const handleTagAuthority = (selected: string) => {
    setAuthorityTag(selected);
    setAuthorityDialog(false);
  };

  const toggleComments = () => setShowComments((prev) => !prev); // 👈 TOGGLE FUNCTION

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

  const authorityColors: Record<string, string> = {
    Municipal: "bg-blue-100 text-blue-700 border-blue-300",
    Water: "bg-cyan-100 text-cyan-700 border-cyan-300",
    Police: "bg-indigo-100 text-indigo-700 border-indigo-300",
    Electricity: "bg-amber-100 text-amber-700 border-amber-300",
    Other: "bg-gray-100 text-gray-700 border-gray-300",
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

        <div className="flex gap-2">
          {tag && (
            <Badge
              variant="outline"
              className={`${tagColors[tag] || "bg-primary/5 text-primary border-primary/40"} px-3 py-1 font-medium`}
            >
              {tag}
            </Badge>
          )}

          {authorityTag && (
            <Badge
              variant="outline"
              className={`${authorityColors[authorityTag] || "bg-gray-100"} px-3 py-1 font-medium flex items-center gap-1.5`}
            >
              <Building2 className="h-4 w-4" /> {authorityTag}
            </Badge>
          )}
        </div>
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
        {/* Upvote */}
        <Popover open={showVoteBreakdown} onOpenChange={setShowVoteBreakdown}>
          <PopoverTrigger asChild>
            <button
              aria-label="Upvote post"
              onClick={handleUpvote}
              className="flex items-center gap-2 text-muted-foreground hover:text-primary transition-all duration-200 hover:scale-110 group/vote"
            >
              <ArrowUp className="h-5 w-5 group-hover/vote:translate-y-[-2px] transition-transform duration-200" />
              <span className="text-sm font-semibold">{votes}</span>
            </button>
          </PopoverTrigger>
          <PopoverContent className="w-64 p-0 shadow-xl border-border/60" align="start">
            <div className="p-5 space-y-4">
              <h4 className="font-bold text-sm text-foreground">Vote Breakdown</h4>
              <div className="space-y-3">
                {Object.entries(voteBreakdown).map(([key, value]) => (
                  <div
                    key={key}
                    className="flex items-center justify-between hover:bg-muted/30 -mx-2 px-2 py-1.5 rounded-md transition-colors"
                  >
                    <span className="capitalize text-sm text-muted-foreground font-medium">{key}</span>
                    <Button variant="secondary" size="sm" className="h-7 px-4 shadow-md">
                      {value}
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          </PopoverContent>
        </Popover>

        {/* Comments */}
        <button
          aria-label="View comments"
          onClick={toggleComments} // 👈 toggles the comment section
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
              className="flex items-center gap-2 text-muted-foreground hover:text-primary transition-all duration-200 ml-auto hover:scale-110"
            >
              <MoreVertical className="h-5 w-5" />
            </button>
          </DropdownMenuTrigger>

          <DropdownMenuContent align="end" className="shadow-xl border-border/60">
            {/* Share */}
            <DropdownMenuItem className="cursor-pointer hover:bg-muted/50 transition-colors">
              <Share2 className="h-4 w-4 mr-2" />
              Share
            </DropdownMenuItem>

            {/* Tag Authority */}
            <Dialog open={authorityDialog} onOpenChange={setAuthorityDialog}>
              <DialogTrigger asChild>
                <DropdownMenuItem
                  className="cursor-pointer hover:bg-muted/50 transition-colors"
                  onSelect={(e) => e.preventDefault()}
                >
                  <Building2 className="h-4 w-4 mr-2" />
                  Tag Authority
                </DropdownMenuItem>
              </DialogTrigger>

              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Tag Authority</DialogTitle>
                </DialogHeader>
                <p className="text-sm text-muted-foreground mb-3">
                  Choose the department best suited to handle this issue.
                </p>

                <RadioGroup onValueChange={handleTagAuthority}>
                  {["Municipal", "Water", "Electricity", "Police", "Other"].map((dept) => (
                    <div key={dept} className="flex items-center space-x-3">
                      <RadioGroupItem value={dept} id={dept} />
                      <Label htmlFor={dept}>{dept} Department</Label>
                    </div>
                  ))}
                </RadioGroup>

                <DialogFooter className="mt-4 flex justify-end gap-2">
                  <Button variant="outline" onClick={() => setAuthorityDialog(false)}>
                    Cancel
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

            {/* Report */}
            <Dialog open={reportDialog} onOpenChange={setReportDialog}>
              <DialogTrigger asChild>
                <DropdownMenuItem
                  className="cursor-pointer text-danger hover:bg-danger/10 transition-colors"
                  onSelect={(e) => e.preventDefault()}
                >
                  <Flag className="h-4 w-4 mr-2" />
                  {isReported ? "Reported" : "Report"}
                </DropdownMenuItem>
              </DialogTrigger>

              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Report Post</DialogTitle>
                </DialogHeader>
                <p className="text-sm text-muted-foreground">
                  Are you sure you want to report this post? It will be reviewed by moderators or concerned authorities.
                </p>
                <DialogFooter className="mt-4 flex justify-end gap-2">
                  <Button variant="outline" onClick={() => setReportDialog(false)}>
                    Cancel
                  </Button>
                  <Button variant="destructive" onClick={handleReport}>
                    Confirm Report
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </DropdownMenuContent>
        </DropdownMenu>

        {isReported && (
          <span className="flex items-center gap-1.5 text-sm text-red-600 ml-2">
            <CheckCircle2 className="h-4 w-4" /> Reported
          </span>
        )}
      </div>

      {/* Comment Section (Shown only when clicked) */}
      {showComments && (
        <div className="mt-3">
          <CommentSection />
        </div>
      )}
    </article>
  );
};
