// ✅ Shared Post interface

export interface Post {
  _id: string;
  title: string;
  description: string;
  category: string;        // e.g. "Municipal", "Water", "Police"
  location: string;        // "Jaypee University, Noida, UP"
  image?: string;
  createdAt: string;

  user?: {
    _id?: string;
    username?: string;
    email?: string;
  };

  upvotes?: number;
  comments?: any[];
  isUpvoted?: boolean;
}
