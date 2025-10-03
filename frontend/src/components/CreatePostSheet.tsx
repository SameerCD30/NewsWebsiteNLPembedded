import { useState } from "react";
import { Plus, X, Upload } from "lucide-react";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "./ui/sheet";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Textarea } from "./ui/textarea";
import { Label } from "./ui/label";

export const CreatePostSheet = () => {
  const [open, setOpen] = useState(false);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [location, setLocation] = useState("");
  const [image, setImage] = useState<string | null>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = () => {
    // Handle post submission
    console.log({ title, description, location, image });
    setOpen(false);
    // Reset form
    setTitle("");
    setDescription("");
    setLocation("");
    setImage(null);
  };

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <button className="fixed left-8 top-24 z-40 flex items-center gap-3 rounded-full bg-primary px-8 py-3.5 text-primary-foreground shadow-xl hover:shadow-2xl hover:bg-primary/90 hover:scale-105 transition-all duration-300 font-semibold group">
          <Plus className="h-5 w-5 group-hover:rotate-90 transition-transform duration-300" />
          <span>Create Post</span>
        </button>
      </SheetTrigger>
      <SheetContent side="left" className="w-[400px] sm:w-[540px] shadow-2xl overflow-y-auto">
        <SheetHeader className="pb-6 border-b border-border/50">
          <SheetTitle className="text-2xl font-bold">Create New Post</SheetTitle>
        </SheetHeader>
        
        <div className="space-y-6 mt-8">
          <div className="space-y-3">
            <Label htmlFor="title" className="text-sm font-semibold text-foreground">Title</Label>
            <Input
              id="title"
              placeholder="Brief title for your issue"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="h-11 border-border/60 focus:border-primary focus:ring-primary/20 transition-all"
            />
          </div>

          <div className="space-y-3">
            <Label htmlFor="description" className="text-sm font-semibold text-foreground">Description</Label>
            <Textarea
              id="description"
              placeholder="Describe the issue in detail..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="min-h-[140px] border-border/60 focus:border-primary focus:ring-primary/20 transition-all resize-none"
            />
          </div>

          <div className="space-y-3">
            <Label htmlFor="location" className="text-sm font-semibold text-foreground">Location</Label>
            <Input
              id="location"
              placeholder="Where is this issue located?"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
              className="h-11 border-border/60 focus:border-primary focus:ring-primary/20 transition-all"
            />
          </div>

          <div className="space-y-3">
            <Label htmlFor="image" className="text-sm font-semibold text-foreground">Add Image</Label>
            <div>
              {image ? (
                <div className="relative overflow-hidden rounded-xl shadow-lg group">
                  <img
                    src={image}
                    alt="Upload preview"
                    className="w-full h-56 object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                  <button
                    onClick={() => setImage(null)}
                    className="absolute top-3 right-3 bg-background/90 backdrop-blur-sm p-2 rounded-full hover:bg-danger hover:text-danger-foreground transition-all shadow-lg"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              ) : (
                <label
                  htmlFor="image"
                  className="flex flex-col items-center justify-center w-full h-56 border-2 border-dashed border-border/60 rounded-xl cursor-pointer hover:border-primary/50 hover:bg-primary/5 transition-all group"
                >
                  <Upload className="h-12 w-12 text-muted-foreground group-hover:text-primary group-hover:scale-110 transition-all duration-200 mb-3" />
                  <span className="text-sm font-medium text-muted-foreground group-hover:text-primary transition-colors">
                    Click to upload image
                  </span>
                  <input
                    id="image"
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleImageUpload}
                  />
                </label>
              )}
            </div>
          </div>

          <div className="flex gap-3 pt-6">
            <Button
              variant="danger"
              onClick={handleSubmit}
              className="flex-1 h-12 text-base font-semibold shadow-lg hover:shadow-xl transition-all duration-200"
            >
              Submit Post
            </Button>
            <Button
              variant="outline"
              onClick={() => setOpen(false)}
              className="flex-1 h-12 text-base font-medium hover:bg-muted/50 transition-all"
            >
              Cancel
            </Button>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
};
