/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#071019",
        panel: "#101922",
        line: "#223244",
        cyanfire: "#18d4ff"
      }
    }
  },
  plugins: []
};

