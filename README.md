# Furniture Recommendation Application

This project is a furniture recommendation application that provides users with personalized furniture suggestions based on their preferences and needs. The application is structured into four main components: backend, frontend, notebooks, and data.

## Project Structure

```
furniture-recommender
├── backend              # Backend application
│   ├── app             # Main application code
│   ├── requirements.txt # Python dependencies
│   ├── Dockerfile      # Docker configuration
│   └── README.md       # Backend documentation
├── frontend             # Frontend application
│   ├── src             # Source code for the frontend
│   ├── package.json    # Frontend dependencies
│   ├── tsconfig.json   # TypeScript configuration
│   └── README.md       # Frontend documentation
├── notebooks            # Jupyter notebooks for analysis and modeling
│   ├── 01-exploration.ipynb # Exploratory data analysis
│   └── 02-modeling.ipynb    # Modeling process
├── data                 # Data storage
│   ├── raw             # Raw data files
│   ├── processed       # Processed data files
│   └── README.md       # Data documentation
├── docker-compose.yml   # Docker Compose configuration
├── .gitignore           # Git ignore file
└── README.md            # Project overview and documentation
```

## Getting Started

### Prerequisites

- Python 3.x
- Node.js and npm
- Docker (optional, for containerization)

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd furniture-recommender
   ```

2. Set up the backend:
   - Navigate to the `backend` directory.
   - Install the required Python packages:
     ```
     pip install -r requirements.txt
     ```

3. Set up the frontend:
   - Navigate to the `frontend` directory.
   - Install the required npm packages:
     ```
     npm install
     ```

### Running the Application

- To run the backend, execute:
  ```
  python app/main.py
  ```

- To run the frontend, execute:
  ```
  npm start
  ```

### Usage

- Access the frontend application in your web browser at `http://localhost:3000`.
- The backend API can be accessed at `http://localhost:8000/api/v1`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.