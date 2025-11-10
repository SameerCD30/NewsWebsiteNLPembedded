import { useEffect, useState } from "react";
import { Header } from "@/components/Header";
import { CreatePostSheet } from "@/components/CreatePostSheet";
import { FilterControl } from "@/components/FilterControl";
import { PostCard } from "@/components/PostCard";
import { fetchPosts } from "@/api/api";

interface Post {
  _id: string;
  title: string;
  description: string;
  category: string;
  location: string;
  image?: string;
  createdAt: string;
  user?: {
    username?: string;
    email?: string;
  };
  upvotes?: number;
  comments?: any[];
}

const Index = () => {
  const [posts, setPosts] = useState<Post[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const loadPosts = async () => {
      try {
        const res = await fetchPosts();
        console.log("Fetched posts:", res.data);
        setPosts(res.data);
      } catch (err: any) {
        console.error(" Error fetching posts:", err);
        setError("Failed to load posts. Please try again later.");
      } finally {
        setLoading(false);
      }
    };
    loadPosts();
  }, []);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <Header />
      <CreatePostSheet />

      <main className="pt-16 min-h-screen">
        <div className="max-w-4xl mx-auto px-8 py-12">
          {/* Filter Section */}
          <FilterControl />

          {/* Status messages */}
          {loading ? (
            <p className="text-gray-400 text-center mt-10 animate-pulse">
              Loading posts...
            </p>
          ) : error ? (
            <p className="text-red-500 text-center mt-10">{error}</p>
          ) : posts.length === 0 ? (
            <p className="text-gray-500 text-center mt-10">
              No issues reported yet. Be the first to upload!
            </p>
          ) : (
            <div className="space-y-8">
              {posts.map((post) => (
                <PostCard key={post._id} post={post} />
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default Index;
