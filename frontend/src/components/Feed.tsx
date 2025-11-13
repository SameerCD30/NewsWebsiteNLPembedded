import { useEffect, useState } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Post } from "@/types/post";
import { Location } from "@/types/location";
import { PostCard } from "@/components/PostCard";
import ChangeLocationModal from "@/components/ChangeLocationModal";
import Header from "@/components/Header";
import { useToast } from "@/components/ui/use-toast";
import { CreatePostSheet } from "@/components/CreatePostSheet";

const Feed: React.FC = () => {
  const [scope, setScope] = useState<"local" | "state" | "national">("local");

  const [location, setLocation] = useState<Location>(() => {
    const saved = localStorage.getItem("userLocation");
    return saved
      ? JSON.parse(saved)
      : { city: "Noida", state: "Uttar Pradesh", country: "India" };
  });

  const [posts, setPosts] = useState<Post[]>([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [loading, setLoading] = useState(false);

  const { toast } = useToast();

  useEffect(() => {
    fetchPosts();
  }, [scope, location]);

  const fetchPosts = async () => {
    try {
      setLoading(true);
      const res = await axios.get<Post[]>("http://localhost:8081/api/posts", {
        params: { scope, city: location.city, state: location.state },
      });

      setPosts(res.data);
    } catch (err) {
      console.error("Error fetching posts:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleLocationChange = (newLoc: Location) => {
    setLocation(newLoc);
    localStorage.setItem("userLocation", JSON.stringify(newLoc));

    toast({
      title: "📍 Location Updated",
      description: `${newLoc.city}, ${newLoc.state} | ${newLoc.country}`,
      duration: 3000,
    });
  };

  return (
    <>
      <Header onChangeLocation={() => setIsModalOpen(true)} />

      <main className="pt-20 px-4 md:px-8 flex flex-col gap-6 max-w-4xl mx-auto w-full bg-[#121212]/20 rounded-2xl shadow-inner">

        {/* Centered Tabs + Location */}
        <motion.div
          initial={{ y: -10, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.4 }}
          className="flex flex-col items-center gap-3 pt-4"
        >
          {/* Tabs Centered */}
          <div className="w-full flex justify-center">
            <Tabs value={scope} onValueChange={(val) => setScope(val as any)}>
              <TabsList className="bg-zinc-900 px-2 py-1 rounded-full flex gap-1">
                
                <TabsTrigger
                  value="local"
                  className="px-5 py-1.5 rounded-full text-sm font-medium
                    transition-all duration-200
                    data-[state=active]:bg-orange-600
                    data-[state=active]:text-white
                    data-[state=inactive]:text-gray-300"
                >
                  Local
                </TabsTrigger>

                <TabsTrigger
                  value="state"
                  className="px-5 py-1.5 rounded-full text-sm font-medium
                    transition-all duration-200
                    data-[state=active]:bg-orange-600
                    data-[state=active]:text-white
                    data-[state=inactive]:text-gray-300"
                >
                  State
                </TabsTrigger>

                <TabsTrigger
                  value="national"
                  className="px-5 py-1.5 rounded-full text-sm font-medium
                    transition-all duration-200
                    data-[state=active]:bg-orange-600
                    data-[state=active]:text-white
                    data-[state=inactive]:text-gray-300"
                >
                  National
                </TabsTrigger>

              </TabsList>
            </Tabs>
          </div>

          {/* Location */}
          <motion.p
            key={JSON.stringify(location)}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4 }}
            className="text-sm text-muted-foreground tracking-wide"
          >
            📍 {location.city}, {location.state} | {location.country}
          </motion.p>
        </motion.div>

        {/* Feed */}
        <AnimatePresence mode="wait">
          {loading ? (
            <motion.p
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="text-center text-muted-foreground py-10"
            >
              Loading posts…
            </motion.p>
          ) : posts.length > 0 ? (
            <motion.div
              key="feed"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="grid gap-6"
            >
              {posts.map((post) => (
                <motion.div
                  key={post._id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4 }}
                >
                  <PostCard post={post} />
                </motion.div>
              ))}
            </motion.div>
          ) : (
            <motion.p
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="text-center text-muted-foreground py-10"
            >
              No posts found for this area.
            </motion.p>
          )}
        </AnimatePresence>

        {/* Modals */}
        <ChangeLocationModal
          open={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          onSave={handleLocationChange}
          currentLocation={location}
        />

        <CreatePostSheet />
      </main>
    </>
  );
};

export default Feed;
