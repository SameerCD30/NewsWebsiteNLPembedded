import { useToast } from "@/hooks/use-toast";
import { createPost } from "../api/api";
import { MapPicker } from "@/components/MapPicker";
import { useState } from "react";
import { Plus, X, Upload } from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "./ui/sheet";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Textarea } from "./ui/textarea";
import { Label } from "./ui/label";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "./ui/select";

export const CreatePostSheet = () => {
  const { toast } = useToast();

  const [open, setOpen] = useState(false);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [location, setLocation] = useState<{
    lat?: number;
    lng?: number;
    address?: string;
    city?: string;
    state?: string;
    country?: string;
    pincode?: string;
    landmark?: string;
  } | null>(null);
  const [image, setImage] = useState<string | null>(null);
  const [tagAuthority, setTagAuthority] = useState("");
  const [showMap, setShowMap] = useState(false);

  const handleOpenClick = () => {
    const token = localStorage.getItem("token");
    if (!token) {
      toast({
        title: "Login Required",
        description: "Please log in to create a post.",
        variant: "destructive",
      });
      return;
    }
    setOpen(true);
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => setImage(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async () => {
    if (!title || !description || !location?.address || !tagAuthority) {
      toast({
        title: "Missing Fields",
        description: "Please fill in all required fields.",
        variant: "destructive",
      });
      return;
    }

    try {
      // Extract pincode
      let pincode = location?.pincode;
      if (!pincode && location?.address) {
        const match = location.address.match(/\b\d{6}\b/);
        pincode = match ? match[0] : "";
      }

      const postData = {
        title,
        description,
        category: tagAuthority,
        image,
        location: {
          address: `${location.address}${location.landmark ? ", " + location.landmark : ""}`,
          lat: location.lat,
          lng: location.lng,
          city: location.city,
          state: location.state,
          country: "India",
          pincode,
        },
      };

      const res = await createPost(postData);
      console.log("Post created:", res.data);

      toast({
        title: "🎉 Issue Submitted",
        description: "Your post is now live for your city!",
      });

      // Reset
      setOpen(false);
      setTitle("");
      setDescription("");
      setLocation(null);
      setImage(null);
      setTagAuthority("");

    } catch (error: any) {
      console.error("Error creating post:", error);

      toast({
        title: "⚠️ Post Rejected",
        description:
          error.response?.data?.message ||
          "This post doesn't appear to be a valid civic issue.",
        variant: "destructive",
      });
    }
  };

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <button
          onClick={handleOpenClick}
          className="fixed top-24 left-10 z-50 flex items-center gap-3 rounded-full 
            bg-red-600 px-6 py-3 text-white font-semibold shadow-lg 
            hover:bg-red-700 hover:scale-105 transition-all duration-300 group"
        >
          <Plus className="h-5 w-5 group-hover:rotate-90 transition-transform duration-300" />
          <span>Create Post</span>
        </button>
      </SheetTrigger>

      <SheetContent
        side="left"
        className="w-[400px] sm:w-[540px] overflow-y-auto border-r border-red-700/40
          bg-gradient-to-br from-[#161616] via-[#1c1c1c] to-[#111111]
          shadow-[0_0_45px_rgba(255,0,0,0.2)] backdrop-blur-lg text-gray-100"
      >
        <SheetHeader className="pb-6 border-b border-red-900/40">
          <SheetTitle className="text-2xl font-bold text-white drop-shadow-md">
            Create New Post
          </SheetTitle>
          <p className="text-sm text-gray-400 leading-relaxed pt-1">
            Share your grievance with the community. Tag the related department
            to help resolve it faster.
          </p>
        </SheetHeader>

        <div className="space-y-6 mt-8">

          {/* TITLE */}
          <div className="space-y-3">
            <Label className="text-sm font-semibold text-gray-300">
              Title
            </Label>
            <Input
              placeholder="Brief title for your issue"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="h-11 border border-gray-700 bg-[#1b1b1b]"
            />
          </div>

          {/* DESCRIPTION */}
          <div className="space-y-3">
            <Label className="text-sm font-semibold text-gray-300">
              Description
            </Label>
            <Textarea
              placeholder="Describe the issue in detail..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="min-h-[140px] border border-gray-700 bg-[#1b1b1b]"
            />
          </div>

          {/* LOCATION */}
          <div className="space-y-3">
            <Label className="text-sm font-semibold text-gray-300">
              Location
            </Label>

            {!showMap && !location ? (
              <Button
                variant="outline"
                onClick={() => setShowMap(true)}
                className="w-full border border-gray-700 text-gray-200"
              >
                📍 Select on Map
              </Button>
            ) : showMap ? (
              <MapPicker
                onSelect={(loc) => {
                  setLocation(loc);
                  setShowMap(false);
                }}
              />
            ) : (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <p className="text-sm text-gray-400 flex-1 truncate">
                    {location?.address}
                  </p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowMap(true)}
                    className="border border-gray-700 text-gray-300"
                  >
                    Change
                  </Button>
                </div>
                <Input
                  placeholder="Add landmark (optional)"
                  value={location?.landmark || ""}
                  onChange={(e) =>
                    setLocation((prev) => ({ ...prev, landmark: e.target.value }))
                  }
                  className="h-11 border border-gray-700 bg-[#1b1b1b]"
                />
              </div>
            )}
          </div>

          {/* TAG AUTHORITY */}
          <div className="space-y-3">
            <Label className="text-sm font-semibold text-gray-300">
              Tag Authority
            </Label>
            <Select value={tagAuthority} onValueChange={setTagAuthority}>
              <SelectTrigger className="w-full h-11 border border-gray-700 bg-[#1b1b1b] text-gray-100">
                <SelectValue placeholder="Select authority" />
              </SelectTrigger>
              <SelectContent className="bg-[#1c1c1c] border border-gray-700 text-gray-200">
                <SelectItem value="Municipal">🏙 Municipal</SelectItem>
                <SelectItem value="Water">💧 Water Dept</SelectItem>
                <SelectItem value="Electricity">⚡ Electricity</SelectItem>
                <SelectItem value="Police">🚔 Police</SelectItem>
                <SelectItem value="Other">🧾 Other</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* IMAGE UPLOAD */}
          <div className="space-y-3">
            <Label className="text-sm font-semibold text-gray-300">
              Add Image
            </Label>
            <div>
              {image ? (
                <div className="relative overflow-hidden rounded-xl shadow-lg group">
                  <img
                    src={image}
                    alt="Preview"
                    className="w-full h-56 object-cover"
                  />
                  <button
                    onClick={() => setImage(null)}
                    className="absolute top-3 right-3 bg-black/60 p-2 rounded-full"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              ) : (
                <label className="flex flex-col items-center justify-center w-full h-56 border-2 border-dashed border-gray-700 
                  rounded-xl cursor-pointer hover:border-red-600/50 hover:bg-red-600/5 transition"
                >
                  <Upload className="h-12 w-12 text-gray-500 mb-3" />
                  <span className="text-sm text-gray-400">Click to upload image</span>
                  <input
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleImageUpload}
                  />
                </label>
              )}
            </div>
          </div>

          {/* SUBMIT */}
          <div className="flex gap-3 pt-6">
            <Button
              variant="danger"
              onClick={handleSubmit}
              className="flex-1 h-12 bg-red-600 hover:bg-red-700"
            >
              Submit Post
            </Button>
            <Button
              variant="outline"
              onClick={() => setOpen(false)}
              className="flex-1 h-12 border border-gray-700"
            >
              Cancel
            </Button>
          </div>

        </div>
      </SheetContent>
    </Sheet>
  );
};
