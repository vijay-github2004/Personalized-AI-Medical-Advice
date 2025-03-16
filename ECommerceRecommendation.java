import java.util.*;

class Product {
    int id;
    String name;
    String category;
    double price;

    public Product(int id, String name, String category, double price) {
        this.id = id;
        this.name = name;
        this.category = category;
        this.price = price;
    }
}

class User {
    int id;
    String name;
    Map<Integer, Integer> ratings = new HashMap<>(); // Product ID -> Rating
    List<Integer> purchaseHistory = new ArrayList<>();

    public User(int id, String name) {
        this.id = id;
        this.name = name;
    }

    public void rateProduct(int productId, int rating) {
        ratings.put(productId, rating);
    }

    public void purchaseProduct(int productId) {
        purchaseHistory.add(productId);
    }
}

class ECommerceSystem {
    List<User> users = new ArrayList<>();
    List<Product> products = new ArrayList<>();
    Map<Integer, String> reviews = new HashMap<>(); // Product ID -> Review

    public void addUser(User user) {
        users.add(user);
    }

    public void addProduct(Product product) {
        products.add(product);
    }

    public User findUserById(int userId) {
        for (User user : users) {
            if (user.id == userId) return user;
        }
        return null;
    }

    public List<Product> recommendProducts(User user) {
        List<Product> recommended = new ArrayList<>();
        for (Product product : products) {
            if (!user.purchaseHistory.contains(product.id)) {
                recommended.add(product);
            }
        }
        return recommended;
    }

    public void addReview(int productId, String review) {
        reviews.put(productId, review);
    }
}

public class ECommerceRecommendation {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        ECommerceSystem system = new ECommerceSystem();

        while (true) {
            System.out.println("\n1. Add User");
            System.out.println("2. Add Product");
            System.out.println("3. Rate Product");
            System.out.println("4. Purchase Product");
            System.out.println("5. Get Recommendations");
            System.out.println("6. Add Review");
            System.out.println("7. Exit");
            System.out.print("Enter your choice: ");
            
            int choice = scanner.nextInt();
            scanner.nextLine();

            switch (choice) {
                case 1:
                    System.out.print("Enter User ID: ");
                    int userId = scanner.nextInt();
                    scanner.nextLine();
                    System.out.print("Enter User Name: ");
                    String userName = scanner.nextLine();
                    system.addUser(new User(userId, userName));
                    System.out.println("User added successfully!");
                    break;

                case 2:
                    System.out.print("Enter Product ID: ");
                    int productId = scanner.nextInt();
                    scanner.nextLine();
                    System.out.print("Enter Product Name: ");
                    String productName = scanner.nextLine();
                    System.out.print("Enter Category: ");
                    String category = scanner.nextLine();
                    System.out.print("Enter Price: ");
                    double price = scanner.nextDouble();
                    system.addProduct(new Product(productId, productName, category, price));
                    System.out.println("Product added successfully!");
                    break;

                case 3:
                    System.out.print("Enter User ID: ");
                    User user = system.findUserById(scanner.nextInt());
                    if (user == null) {
                        System.out.println("User not found!");
                        break;
                    }
                    System.out.print("Enter Product ID to Rate: ");
                    int pId = scanner.nextInt();
                    System.out.print("Enter Rating (1-5): ");
                    user.rateProduct(pId, scanner.nextInt());
                    System.out.println("Rating added successfully!");
                    break;

                case 4:
                    System.out.print("Enter User ID: ");
                    user = system.findUserById(scanner.nextInt());
                    if (user == null) {
                        System.out.println("User not found!");
                        break;
                    }
                    System.out.print("Enter Product ID to Purchase: ");
                    user.purchaseProduct(scanner.nextInt());
                    System.out.println("Product purchased successfully!");
                    break;

                case 5:
                    System.out.print("Enter User ID for Recommendations: ");
                    user = system.findUserById(scanner.nextInt());
                    if (user == null) {
                        System.out.println("User not found!");
                        break;
                    }
                    List<Product> recommendations = system.recommendProducts(user);
                    System.out.println("Recommended Products:");
                    for (Product p : recommendations) {
                        System.out.println(p.name + " - $" + p.price);
                    }
                    break;

                case 6:
                    System.out.print("Enter Product ID to Review: ");
                    int reviewProductId = scanner.nextInt();
                    scanner.nextLine();
                    System.out.print("Enter Review: ");
                    system.addReview(reviewProductId, scanner.nextLine());
                    System.out.println("Review added successfully!");
                    break;

                case 7:
                    System.out.println("Exiting... Thank you!");
                    scanner.close();
                    return;

                default:
                    System.out.println("Invalid choice!");
                    break;
            }
        }
    }
}
