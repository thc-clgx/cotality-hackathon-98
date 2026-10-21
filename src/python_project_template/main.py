import sys
import os


def main():
    # Print the script's name
    print(f"Script: {sys.argv[0]}")

    # Print the script's arguments
    print("Arguments:")
    for arg in sys.argv[1:]:
        print(f"- {arg}")

    # Print the environment variables
    print("\nEnvironment Variables:")
    for key, value in os.environ.items():
        print(f"- {key}: {value}")
    sys.exit(0)


if __name__ == "__main__":
    sys.exit(main())
