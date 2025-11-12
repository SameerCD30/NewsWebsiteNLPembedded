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
        params: {
          scope,
          city: location.city,
          state: location.state,
        },
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
      description: `${newLoc.city}, ${newLoc.state}, ${newLoc.country}`,
      duration: 3000,
    });
  };

  return (
    <>
      <Header onChangeLocation={() => setIsModalOpen(true)} />

      <main className="pt-20 px-4 md:px-8 flex flex-col gap-6 max-w-5xl mx-auto w-full bg-[#121212]/30 rounded-2xl shadow-inner">
        {/* Tabs + Location Display */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="flex justify-between items-center flex-wrap gap-3 mt-4"
        >

          <Tabs
            value={scope}
            onValueChange={(val) =>
              setScope(val as "local" | "state" | "national")
            }
          >
            <TabsList className="bg-zinc-900 p-1 rounded-full">
              <TabsTrigger value="local" className="px-4 py-1.5">
                Local
              </TabsTrigger>
              <TabsTrigger value="state" className="px-4 py-1.5">
                State
              </TabsTrigger>
              <TabsTrigger value="national" className="px-4 py-1.5">
                National
              </TabsTrigger>
            </TabsList>
          </Tabs>

          <motion.p
            key={JSON.stringify(location)}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4 }}
            className="text-sm text-muted-foreground"
          >
            📍 {location.city}, {location.state}, {location.country}
          </motion.p>
        </motion.div>

        {/* Feed Grid */}
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
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.4 }}
              className="grid gap-5"
            >
              {posts.map((post) => (
                <motion.div
                  key={post._id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, delay: 0.05 }}
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
              transition={{ duration: 0.4 }}
              className="text-center text-muted-foreground py-10"
            >
              No posts found for this area.
            </motion.p>
          )}
        </AnimatePresence>

        {/* Location Modal */}
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
