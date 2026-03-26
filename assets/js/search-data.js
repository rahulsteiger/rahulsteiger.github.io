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
        },{id: "post-an-evaluation-of-deepspeed-zero-compilation-and-offloading",
      
        title: "An Evaluation of DeepSpeed ZeRO, Compilation, and Offloading",
      
      description: "Notes from the final project of the course Large-Scale AI Engineering.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/notes/2025/lsaie-project/";
        
      },
    },{id: "post-cuda-programming-optimizing-gemm",
      
        title: "CUDA Programming - Optimizing GEMM",
      
      description: "Notes on me (attempting) to create a fast GEMM CUDA implementation.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/notes/2025/cuda-programming_gemm/";
        
      },
    },{id: "post-cuda-programming-fundamentals",
      
        title: "CUDA Programming - Fundamentals",
      
      description: "Notes on me (properly) learning how to use CUDA.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/notes/2025/cuda-programming/";
        
      },
    },{id: "post-running-deepseek-r1-locally",
      
        title: "Running DeepSeek R1 locally",
      
      description: "Steps required to run inference for DeepSeek R1 using llama.cpp on a single HPC node equipped with 4 A100 GPUs and 1 TB of memory.",
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
      
      description: "Ensemble methods combine multiple simple learning algorithms to achieve superior overall performance. This note is an adaptation of a group project from the CS4270 course I took at the National University of Singapore during my exchange.",
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
            },},{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
