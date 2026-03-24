import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig(function (_a) {
    var mode = _a.mode;
    var env = loadEnv(mode, ".", "");
    var backendHttp = env.BACKEND_URL || "http://127.0.0.1:8001";
    var backendWs = backendHttp.replace(/^http/, "ws");
    return {
        plugins: [react()],
        server: {
            host: "0.0.0.0",
            port: 5173,
            proxy: {
                "/api": {
                    target: backendHttp,
                    changeOrigin: true
                },
                "/ws": {
                    target: backendWs,
                    ws: true,
                    changeOrigin: true
                }
            }
        }
    };
});
