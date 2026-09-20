"""
===============================================================================
DUMMY DATA                                                      [DEMO data]
===============================================================================

Seeds the SQLite database with a realistic spread of Bengaluru food-delivery
scenarios. The data is deliberately chosen so that every branch of the action
pipeline and the escalation policy layer has something to fire on:

  ORD-88213  delivered 3h ago, good customer      -> small refund, auto-approved
  ORD-88455  delivered 5h ago, 1240              -> over supervisor threshold
  ORD-88601  delivered 2h ago, serial refunder   -> fraud-review escalation
  ORD-88777  delivered 126h ago                  -> outside the refund window
  ORD-88912  out for delivery, partner moving    -> no refund, share tracking
  ORD-88990  already refunded                    -> double-refund denial
  ORD-89011  placed 4 min ago, preparing         -> cancellable, instructions OK
  ORD-89044  out for delivery, partner STUCK 23m -> DP_MOVEMENT_ISSUE upheld
  ORD-89077  out for delivery, running late      -> late but partner moving
  ORD-89100  delivered 1h ago, wallet payment    -> instant refund path

Run `python -m zomato_support.db.seed` to rebuild.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path

from . import get_db, init_db

CUSTOMERS = [
    # (id, name, email, last4, city, since, gold, trust, orders, refunds90d, notes)
    ("CUS-1001", "Ananya Rao", "ananya.rao@example.com", "4417", "Bengaluru",
     "2021-04-12", 1, 0.91, 312, 1,
     "Long-tenured Gold member. No disputed claims on record."),
    ("CUS-1002", "Vikram Sethi", "vikram.sethi@example.com", "8802", "Bengaluru",
     "2023-09-01", 0, 0.74, 58, 1,
     "One prior refund for a genuinely missing item, verified by restaurant."),
    ("CUS-1003", "Rohit Malhotra", "rohit.malhotra@example.com", "2290", "Bengaluru",
     "2025-11-20", 0, 0.28, 14, 5,
     "Five refund claims in 90 days across different restaurants. Two prior photo "
     "submissions flagged by image forensics. Account under Trust & Safety review."),
    ("CUS-1004", "Priya Nair", "priya.nair@example.com", "6135", "Bengaluru",
     "2024-02-17", 1, 0.88, 141, 0, "No refund history."),
    ("CUS-1005", "Arjun Menon", "arjun.menon@example.com", "7713", "Bengaluru",
     "2022-07-30", 1, 0.86, 203, 0, "Frequent office-lunch orders."),
]

# (order_id, customer, restaurant, phone, status, delivered_hours_ago,
#  placed_min_ago, total, payment, ref, address, instructions, cancel_window)
ORDERS = [
    ("ORD-88213", "CUS-1001", "Burger Singh, Koramangala", "+91-80-4711-2201",
     "delivered", 3, 200, 487, "UPI", "PAY-9f21ac",
     "42, 4th Block, Koramangala, Bengaluru 560034", None, 60),
    ("ORD-88455", "CUS-1002", "Bombay Brasserie, Indiranagar", "+91-80-4711-3345",
     "delivered", 5, 320, 1240, "Credit Card", "PAY-3b77de",
     "18, 100 Feet Road, Indiranagar, Bengaluru 560038", "Ring the bell twice", 60),
    ("ORD-88601", "CUS-1003", "Meghana Foods, HSR Layout", "+91-80-4711-5567",
     "delivered", 2, 140, 640, "Zomato Wallet", "PAY-c401aa",
     "7, Sector 2, HSR Layout, Bengaluru 560102", None, 60),
    ("ORD-88777", "CUS-1001", "Chai Point, MG Road", "+91-80-4711-7789",
     "delivered", 126, 7600, 320, "UPI", "PAY-7de110",
     "42, 4th Block, Koramangala, Bengaluru 560034", None, 60),
    ("ORD-88912", "CUS-1004", "Truffles, St. Marks Road", "+91-80-4711-9901",
     "out_for_delivery", None, 38, 815, "UPI", "PAY-51ba09",
     "301, Prestige Towers, Richmond Road, Bengaluru 560025", None, 15),
    ("ORD-88990", "CUS-1002", "Sahib Sindh Sultan, Whitefield", "+91-80-4711-2213",
     "delivered", 9, 560, 410, "Credit Card", "PAY-a19c34",
     "18, 100 Feet Road, Indiranagar, Bengaluru 560038", None, 60),
    # Just placed - the cancellation and delivery-instruction happy paths.
    ("ORD-89011", "CUS-1005", "Rameshwaram Cafe, Indiranagar", "+91-80-4711-4456",
     "preparing", None, 4, 396, "UPI", "PAY-b2c410",
     "9, Defence Colony, Indiranagar, Bengaluru 560038", None, 15),
    # Partner has not moved in 23 minutes - DP_MOVEMENT_ISSUE should be upheld.
    ("ORD-89044", "CUS-1001", "Empire Restaurant, Koramangala", "+91-80-4711-6678",
     "out_for_delivery", None, 62, 528, "UPI", "PAY-d80f31",
     "42, 4th Block, Koramangala, Bengaluru 560034", "Leave at the door", 15),
    # Late, but the partner IS moving - escalation should be REJECTED.
    ("ORD-89077", "CUS-1005", "Nagarjuna, Residency Road", "+91-80-4711-8890",
     "out_for_delivery", None, 47, 742, "Credit Card", "PAY-1f9b02",
     "9, Defence Colony, Indiranagar, Bengaluru 560038", None, 15),
    ("ORD-89100", "CUS-1004", "Corner House, Jayanagar", "+91-80-4711-3312",
     "delivered", 1, 90, 285, "Zomato Wallet", "PAY-e54a77",
     "301, Prestige Towers, Richmond Road, Bengaluru 560025", None, 60),
]

ITEMS = {
    "ORD-88213": [("Uttarakhandi Burger", 1, 229), ("Peri Peri Fries", 1, 139),
                  ("Cold Coffee", 1, 119)],
    "ORD-88455": [("Mutton Biryani (Family Pack)", 1, 749), ("Paneer Tikka", 1, 329),
                  ("Gulab Jamun (4 pc)", 1, 162)],
    "ORD-88601": [("Boneless Chicken Biryani", 2, 560), ("Raita", 1, 80)],
    "ORD-88777": [("Masala Chai (Pack of 4)", 1, 220), ("Samosa", 2, 100)],
    "ORD-88912": [("Ultimate Chicken Cheese Burger", 2, 618), ("Choco Lava Cake", 1, 197)],
    "ORD-88990": [("Dal Bukhara", 1, 340), ("Butter Naan", 2, 70)],
    "ORD-89011": [("Ghee Podi Idli", 1, 189), ("Filter Coffee", 2, 118),
                  ("Kara Bath", 1, 89)],
    "ORD-89044": [("Chicken Kebab", 1, 329), ("Rumali Roti", 3, 199)],
    "ORD-89077": [("Andhra Meals", 2, 598), ("Gongura Chicken", 1, 144)],
    "ORD-89100": [("Death by Chocolate", 1, 189), ("Butterscotch Scoop", 2, 96)],
}

# (order_id, partner, phone, rating, vehicle, km, eta, promised, last_moved, area)
TRACKING = [
    ("ORD-88912", "Firoz Ahmed", "+91-99000-11223", 4.8, "Bike", 2.3, 9, 35, 1,
     "Richmond Circle"),
    # Stationary for 23 minutes: the signal a DP_MOVEMENT_ISSUE claim needs.
    ("ORD-89044", "Sunita Menon", "+91-99000-44556", 4.6, "Scooter", 1.1, 26, 30, 23,
     "Sony World Junction"),
    # Late overall, but moving normally - the claim should NOT be upheld.
    ("ORD-89077", "Deepak Rathore", "+91-99000-77889", 4.9, "Bike", 3.8, 14, 30, 2,
     "Richmond Road"),
]

REFUNDS = [
    ("RFD-55120", "ORD-88990", 410, "Item missing", "completed", "Credit Card",
     "5-7 business days"),
]


def seed_all(path: Path | None = None) -> None:
    """Populate every reference table. Idempotent via INSERT OR REPLACE."""
    with get_db(path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO customers (customer_id, name, email, phone_last4, "
            "city, member_since, gold_member, trust_score, lifetime_orders, "
            "refunds_last_90d, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            CUSTOMERS,
        )
        conn.executemany(
            "INSERT OR REPLACE INTO orders (order_id, customer_id, restaurant, "
            "restaurant_phone, status, delivered_hours_ago, placed_minutes_ago, total, "
            "payment_method, payment_ref, delivery_address, delivery_instructions, "
            "cancellable_until_min) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ORDERS,
        )

        conn.execute("DELETE FROM order_items")
        for order_id, items in ITEMS.items():
            conn.executemany(
                "INSERT INTO order_items (order_id, name, qty, price) VALUES (?,?,?,?)",
                [(order_id, n, q, p) for n, q, p in items],
            )

        conn.executemany(
            "INSERT OR REPLACE INTO delivery_tracking (order_id, partner_name, "
            "partner_phone, partner_rating, vehicle, distance_km, eta_minutes, "
            "promised_eta_min, last_moved_min_ago, current_area, tracking_url) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [(*row, f"https://zoma.to/track/{row[0]}") for row in TRACKING],
        )

        conn.executemany(
            "INSERT OR REPLACE INTO refunds (refund_id, order_id, amount, reason, "
            "status, destination, settles_in) VALUES (?,?,?,?,?,?,?)",
            REFUNDS,
        )


def main() -> None:
    init_db()
    seed_all()
    with get_db() as conn:
        for table in ("customers", "orders", "order_items", "delivery_tracking", "refunds"):
            n = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
            print(f"  {table:<20} {n:>4} rows")


if __name__ == "__main__":
    main()
