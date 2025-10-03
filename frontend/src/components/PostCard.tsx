import { useState } from "react";
import { ArrowUp, ArrowDown, MessageCircle, MoreVertical, Share2, Flag } from "lucide-react";
import { Avatar, AvatarFallback } from "./ui/avatar";
import { Badge } from "./ui/badge";
import { Popover, PopoverContent, PopoverTrigger } from "./ui/popover";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "./ui/dropdown-menu";
import { Button } from "./ui/button";

interface PostCardProps {
  author: string;
  timestamp: string;
  content: string;
  upvotes: number;
  comments: number;
  tag?: string;
  image?: string;
}

export const PostCard = ({ 
  author, 
  timestamp, 
  content, 
  upvotes, 
  comments, 
  tag,
  image
}: PostCardProps) => {
  const [showVoteBreakdown, setShowVoteBreakdown] = useState(false);

  const getInitials = (name: string) => {
    return name
      .split(' ')
      .map(n => n[0])
      .join('')
      .toUpperCase();
  };

  // Mock vote breakdown data
  const voteBreakdown = {
    local: Math.floor(upvotes * 0.5),
    state: Math.floor(upvotes * 0.3),
    national: Math.floor(upvotes * 0.2),
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
          <div className="flex items-center gap-2.5">
            <span className="font-semibold text-foreground text-base">{author}</span>
            <span className="text-muted-foreground/50">•</span>
            <span className="text-sm text-muted-foreground/80">{timestamp}</span>
          </div>
        </div>
        
        {tag && (
          <Badge variant="outline" className="border-primary/40 text-primary bg-primary/5 px-3 py-1 font-medium">
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
            className="w-full h-80 object-cover rounded-xl shadow-lg hover:shadow-2xl transition-shadow duration-300"
          />
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center gap-6 pt-4 border-t border-border/50">
        <Popover open={showVoteBreakdown} onOpenChange={setShowVoteBreakdown}>
          <PopoverTrigger asChild>
            <button className="flex items-center gap-2 text-muted-foreground hover:text-primary transition-all duration-200 hover:scale-110 group/vote">
              <div className="flex items-center gap-1.5">
                <ArrowUp className="h-5 w-5 group-hover/vote:translate-y-[-2px] transition-transform duration-200" />
                <span className="text-sm font-semibold">{upvotes}</span>
                <ArrowDown className="h-5 w-5 group-hover/vote:translate-y-[2px] transition-transform duration-200" />
              </div>
            </button>
          </PopoverTrigger>
          <PopoverContent className="w-64 p-0 shadow-xl border-border/60" align="start">
            <div className="p-5 space-y-4">
              <h4 className="font-bold text-sm text-foreground">Vote Breakdown</h4>
              <div className="space-y-3">
                <div className="flex items-center justify-between group/item hover:bg-muted/30 -mx-2 px-2 py-1.5 rounded-md transition-colors">
                  <span className="text-sm text-muted-foreground font-medium">Local</span>
                  <Button variant="danger" size="sm" className="h-7 px-4 shadow-md">
                    {voteBreakdown.local}
                  </Button>
                </div>
                <div className="flex items-center justify-between group/item hover:bg-muted/30 -mx-2 px-2 py-1.5 rounded-md transition-colors">
                  <span className="text-sm text-muted-foreground font-medium">State</span>
                  <Button variant="danger" size="sm" className="h-7 px-4 shadow-md">
                    {voteBreakdown.state}
                  </Button>
                </div>
                <div className="flex items-center justify-between group/item hover:bg-muted/30 -mx-2 px-2 py-1.5 rounded-md transition-colors">
                  <span className="text-sm text-muted-foreground font-medium">National</span>
                  <Button variant="danger" size="sm" className="h-7 px-4 shadow-md">
                    {voteBreakdown.national}
                  </Button>
                </div>
              </div>
            </div>
          </PopoverContent>
        </Popover>
        
        <button className="flex items-center gap-2 text-muted-foreground hover:text-primary transition-all duration-200 hover:scale-110">
          <MessageCircle className="h-5 w-5" />
          <span className="text-sm font-semibold">{comments}</span>
        </button>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button className="flex items-center gap-2 text-muted-foreground hover:text-primary transition-all duration-200 ml-auto hover:scale-110">
              <MoreVertical className="h-5 w-5" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="shadow-xl border-border/60">
            <DropdownMenuItem className="cursor-pointer hover:bg-muted/50 transition-colors">
              <Share2 className="h-4 w-4 mr-2" />
              Share
            </DropdownMenuItem>
            <DropdownMenuItem className="cursor-pointer text-danger hover:bg-danger/10 transition-colors">
              <Flag className="h-4 w-4 mr-2" />
              Report
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </article>
  );
};
