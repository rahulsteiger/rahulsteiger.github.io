// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "About",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-notes",
          title: "Notes",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/notes/index.html";
          },
        },{id: "nav-",
          title: "",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/repositories/";
          },
        },{id: "post-the-gpu-memory-wall-in-llm-serving",
      
        title: "The GPU Memory Wall in LLM Serving",
      
      description: "Why GPU memory is the bottleneck, and what the GH200 changes.Part 1 of 3 from my Master&#39;s thesis at ETH Zurich.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/notes/2026/llm-serving-background/";
        
      },
    },{id: "post-scaling-llm-training-with-deepspeed",
      
        title: "Scaling LLM Training with DeepSpeed",
      
      description: "Making the most of limited GPU memory with DeepSpeed.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/notes/2025/lsaie-project/";
        
      },
    },{id: "post-cuda-programming-optimizing-gemm",
      
        title: "CUDA Programming - Optimizing GEMM",
      
      description: "A work-in-progress attempt at writing a fast GEMM kernel.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/notes/2025/cuda-programming_gemm/";
        
      },
    },{id: "post-cuda-programming-fundamentals",
      
        title: "CUDA Programming - Fundamentals",
      
      description: "The fundamentals of GPU programming, from first principles.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/notes/2025/cuda-programming/";
        
      },
    },{id: "post-running-deepseek-r1-locally",
      
        title: "Running DeepSeek R1 locally",
      
      description: "Running inference for DeepSeek R1 with llama.cpp on a 4×A100 node.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/notes/2025/llamacpp-deepseek/";
        
      },
    },{id: "post-biased-coin",
      
        title: "Biased Coin",
      
      description: "A nice analytical solution I came up with for a quantitative trading interview question and a valuable life lesson for me.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/notes/2025/biased-coin/";
        
      },
    },{id: "post-ensemble-methods",
      
        title: "Ensemble Methods",
      
      description: "A deep dive into ensemble methods, from decision trees to XGBoost.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/notes/2025/ensemble-methods/";
        
      },
    },{id: "news-a-simple-inline-announcement",
          title: 'A simple inline announcement.',
          description: "",
          section: "News",},{id: "news-a-long-announcement-with-details",
          title: 'A long announcement with details',
          description: "",
          section: "News",handler: () => {
              window.location.href = "/news/announcement_2/";
            },},{id: "news-a-simple-inline-announcement-with-markdown-emoji-sparkles-smile",
          title: 'A simple inline announcement with Markdown emoji! :sparkles: :smile:',
          description: "",
          section: "News",},{id: "projects-project-1",
          title: 'project 1',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project/";
            },},{id: "projects-project-2",
          title: 'project 2',
          description: "a project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/2_project/";
            },},{id: "projects-project-3-with-very-long-name",
          title: 'project 3 with very long name',
          description: "a project that redirects to another website",
          section: "Projects",handler: () => {
              window.location.href = "/projects/3_project/";
            },},{id: "projects-project-4",
          title: 'project 4',
          description: "another without an image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/4_project/";
            },},{id: "projects-project-5",
          title: 'project 5',
          description: "a project with a background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/5_project/";
            },},{id: "projects-project-6",
          title: 'project 6',
          description: "a project with no image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/6_project/";
            },},{id: "projects-project-7",
          title: 'project 7',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/7_project/";
            },},{id: "projects-project-8",
          title: 'project 8',
          description: "an other project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/8_project/";
            },},{id: "projects-project-9",
          title: 'project 9',
          description: "another project with an image 🎉",
          section: "Projects",handler: () => {
              window.location.href = "/projects/9_project/";
            },},];
