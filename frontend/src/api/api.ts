import axios from "axios";

const API = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8081/api",
});

// Automatically add JWT token
API.interceptors.request.use((req) => {
  const token = localStorage.getItem("token");
  if (token) req.headers.Authorization = `Bearer ${token}`;
  return req;
});

// ========================
// Auth endpoints
// ========================
export const loginUser = (data: { email: string; password: string }) =>
  API.post("/auth/login", data);

export const registerUser = (data: { username: string; email: string; password: string }) =>
  API.post("/auth/signup", data);

// ========================
// Post endpoints
// ========================
export const fetchPosts = () => API.get("/posts");
export const createPost = (data: any) => API.post("/posts", data);
export const upvotePost = (postId: string) => API.post(`/posts/${postId}/upvote`);
export const reportPost = (postId: string) => API.post(`/posts/${postId}/report`);
export const addComment = (postId: string, comment: string) =>
  API.post(`/posts/${postId}/comments`, { comment });
