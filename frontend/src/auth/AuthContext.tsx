import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import { api, clearSession, getStoredToken, getStoredUser, storeSession } from "../api/client";
import type { UserOut } from "../api/types";

interface AuthState {
  user: UserOut | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserOut | null>(() => getStoredUser());

  const login = async (email: string, password: string) => {
    const token = await api.login(email, password);
    storeSession(token);
    setUser(token.user);
  };

  const logout = () => {
    clearSession();
    setUser(null);
  };

  const value = useMemo<AuthState>(
    () => ({ user, isAuthenticated: Boolean(user && getStoredToken()), login, logout }),
    [user],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
