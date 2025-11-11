import axios from "axios";


const API = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:5000/api",
});

API.interceptors.request.use((req) => {
  const token = localStorage.getItem("token");
  if (token) req.headers.Authorization = `Bearer ${token}`;
  return req;
});

export const loginUser = (data: { email: string; password: string }) =>
  API.post("/auth/login", data);

export const registerUser = (data: { username: string; email: string; password: string }) =>
  API.post("/auth/signup", data);

export const fetchPosts = () => API.get("/posts");
export const fetchMyPosts = () => API.get("/myposts");
export const fetchCurrentUser = () => API.get("/auth/me");
export const createPost = (data: any) => API.post("/posts", data);
export const upvotePost = (postId: string) => API.post(`/posts/${postId}/upvote`);
export const removeUpvote = (postId: string) => API.post(`/posts/${postId}/unupvote`);
export const reportPost = (postId: string) => API.post(`/posts/${postId}/report`);
export const fetchComments = (postId: string) => API.get(`/posts/${postId}/comments`);
export const addComment = (postId: string, text: string) =>
  API.post(`/posts/${postId}/comments`, { text });

