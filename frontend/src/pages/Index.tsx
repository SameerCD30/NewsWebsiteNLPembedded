import { Header } from "@/components/Header";
import { CreatePostSheet } from "@/components/CreatePostSheet";
import { FilterControl } from "@/components/FilterControl";
import { PostCard } from "@/components/PostCard";

const Index = () => {
  const posts = [
    {
      author: "John Doe",
      timestamp: "2h ago",
      content: "The street light near the community park has been broken for two weeks. It's creating safety concerns for evening walkers.",
      upvotes: 24,
      comments: 2,
      tag: "Potentially fake",
      image: "https://images.unsplash.com/photo-1578678809746-0b29d7d19f1a?w=800&h=600&fit=crop",
    },
    {
      author: "Sarah M",
      timestamp: "5h ago",
      content: "Large pothole causing damage to vehicles. Needs immediate attention before someone gets hurt.",
      upvotes: 18,
      comments: 5,
      tag: "Potentially fake",
      image: "https://images.unsplash.com/photo-1625047509248-ec889cbff17f?w=800&h=600&fit=crop",
    },
    {
      author: "Mike Johnson",
      timestamp: "8h ago",
      content: "The local playground equipment is rusting and becoming unsafe for children. Several parents have raised concerns.",
      upvotes: 32,
      comments: 12,
      image: "https://images.unsplash.com/photo-1587280501635-68a0e82cd5ff?w=800&h=600&fit=crop",
    },
    {
      author: "Emily Chen",
      timestamp: "1d ago",
      content: "Broken traffic signal at Main St intersection causing dangerous situations during rush hour. Please fix urgently!",
      upvotes: 45,
      comments: 8,
      tag: "Potentially fake",
      image: "https://images.unsplash.com/photo-1496247749665-49cf5b1022e9?w=800&h=600&fit=crop",
    },
  ];

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <CreatePostSheet />
      
      <main className="pt-16 min-h-screen">
        <div className="max-w-4xl mx-auto px-8 py-12">
          <FilterControl />
          
          <div className="space-y-8">
            {posts.map((post, index) => (
              <PostCard key={index} {...post} />
            ))}
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;
