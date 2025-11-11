import { useEffect, useState } from "react";
import { Header } from "@/components/Header";
import { PostCard } from "@/components/PostCard";
import { fetchMyPosts, fetchPosts } from "@/api/api";
import { useAuth } from "@/context/AuthContext";

interface Post {
  _id: string;
  title: string;
  description: string;
  category: string;
  location: string;
  image?: string;
  createdAt: string;
   user?: {
    _id?: string;
    username?: string;
    email?: string;
  };
}

const MyPosts = () => {
  const { user } = useAuth();
  const [posts, setPosts] = useState<Post[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const loadUserPosts = async () => {
      try {
        const res = await fetchMyPosts();
        setPosts(res.data);
      } catch (err) {
        console.error("Error loading user posts:", err);
        setError("Failed to load your posts.");
      } finally {
        setLoading(false);
      }
    };

    if (user) loadUserPosts();
  }, [user]);

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="pt-20 max-w-4xl mx-auto px-6">
        <h2 className="text-3xl font-bold text-red-500 mb-8">My Posts</h2>

        {loading ? (
          <p className="text-gray-400">Loading your posts...</p>
        ) : error ? (
          <p className="text-red-500">{error}</p>
        ) : posts.length === 0 ? (
          <p className="text-gray-500">You haven’t posted anything yet.</p>
        ) : (
          <div className="space-y-6">
            {posts.map((post) => (
              <PostCard key={post._id} post={post} />
            ))}
          </div>
        )}
      </main>
    </div>
  );
};

export default MyPosts;
