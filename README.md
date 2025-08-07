# Flowbite Admin Dashboard

A free and open-source admin dashboard template built with Tailwind CSS and Flowbite.

## Table of Contents

- [Flowbite Admin Dashboard](#flowbite-admin-dashboard)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
    - [Development](#development)
    - [Building for Production](#building-for-production)
  - [Architecture](#architecture)
    - [Technologies](#technologies)
    - [Directory Structure](#directory-structure)
  - [Contributing](#contributing)
  - [License](#license)

## Project Overview

This project is a free and open-source admin dashboard template built with modern web technologies. It provides a solid foundation for building admin interfaces, with a focus on developer experience and ease of use.

**Features:**

*   **Responsive Design:** Works on all screen sizes, from mobile to desktop.
*   **Component-based:** Built with reusable components from Flowbite.
*   **Dark Mode:** Includes a dark mode version of the dashboard.
*   **Charts:** Uses ApexCharts for data visualization.
*   **Tooling:** Uses Hugo for static site generation and Webpack for asset bundling.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

*   [Node.js](https://nodejs.org/en/) (v14 or higher)
*   [Hugo](https://gohugo.io/getting-started/installing/) (extended version)
*   [Yarn](https://classic.yarnpkg.com/en/docs/install) (optional, but recommended)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/themesberg/flowbite-admin-dashboard.git
    cd flowbite-admin-dashboard
    ```

2.  **Install dependencies:**

    ```bash
    npm install
    # or
    yarn install
    ```

## Usage

### Development

To start the development server, run the following command:

```bash
npm run start
# or
yarn start
```

This will start a local development server at `http://localhost:1313`. The server will automatically reload when you make changes to the source files.

### Building for Production

To build the project for production, run the following command:

```bash
npm run build
# or
yarn build
```

This will create a `public` directory with the optimized and minified files ready for deployment.

## Architecture

This project follows a modern web development architecture, separating content, layout, and logic.

### Technologies

*   **[Hugo](https://gohugo.io/):** A fast and flexible static site generator written in Go. It is used to build the HTML pages from the content and layout files.
*   **[Tailwind CSS](https://tailwindcss.com/):** A utility-first CSS framework for rapidly building custom designs.
*   **[Flowbite](https://flowbite.com/):** A component library built on top of Tailwind CSS.
*   **[Webpack](https://webpack.js.org/):** A module bundler for JavaScript applications. It is used to bundle the JavaScript source code and its dependencies.
*   **[ApexCharts](https://apexcharts.com/):** A modern charting library for building interactive charts and visualizations.

### Directory Structure

The project has the following directory structure:

```
.
├── content/         # Markdown files for the website content
├── data/            # Data files (JSON, YAML) used by Hugo
├── layouts/         # HTML templates for rendering the content
├── src/             # Source code for JavaScript and CSS
├── static/          # Static assets (images, fonts, etc.)
├── public/          # The output of the build process
├── config.yml       # Hugo configuration file
├── package.json     # Node.js dependencies and scripts
└── webpack.config.js # Webpack configuration file
```

*   `content/`: This directory contains the markdown files that represent the content of the pages. Each file corresponds to a page on the website.
*   `data/`: This directory contains data files in JSON or YAML format. These files can be used in the templates to populate data-driven components.
*   `layouts/`: This directory contains the Hugo templates. The `_default` directory contains the base templates, and other directories can be created for specific content types.
*   `src/`: This directory contains the source code for the frontend assets, including JavaScript and CSS.
*   `static/`: This directory contains static assets that are copied directly to the `public` directory during the build process.
*   `public/`: This directory is where the generated static site is placed after a successful build.
*   `config.yml`: This is the main configuration file for the Hugo project.
*   `package.json`: This file defines the project's dependencies and scripts for building and development.
*   `webpack.config.js`: This file contains the configuration for Webpack, which is used to bundle the JavaScript modules.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or suggestions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
