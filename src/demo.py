from .predict import TicketClassifier


def run_demo():
    clf = TicketClassifier()

    samples = [
        "I was charged twice for my subscription this month. Please refund the extra amount.",
        "The VPN keeps disconnecting every 5 minutes and I cannot access internal tools.",
        "I need to update my leave balance for next month. How do I submit a request?",
        "Can you tell me what your office hours are and where you are located?",
        "URGENT: Production server is DOWN. Customers cannot place orders!",
        "I have a question about your pricing plans for the enterprise tier.",
        "My keyboard is not typing properly after the latest firmware update.",
        "I'd like to request a change in my payroll deduction for health insurance.",
    ]

    print()
    print("=" * 70)
    print("  AUTO EMAIL / TICKET CATEGORIZER -- DEMO")
    print("=" * 70)

    for i, ticket in enumerate(samples, 1):
        result = clf.predict(ticket)

        flag = " !! REVIEW" if result["needs_review"] else " [AUTO]"
        print()
        print(f"  [{i}] {ticket[:80]}{'...' if len(ticket) > 80 else ''}")
        print(f"      -> Category: {result['category']:10s}  "
              f"Conf: {result['confidence']:5.1f}%  "
              f"Priority: {result['priority']:6s}{flag}")

    print()
    print("=" * 70)

    print()
    print("  Interactive mode -- type a ticket (or 'q' to quit):")
    try:
        while True:
            ticket = input("\n  > ").strip()
            if ticket.lower() in ("q", "quit", "exit"):
                break
            if not ticket:
                continue
            result = clf.predict(ticket)
            print(f"  Category  : {result['category']}")
            print(f"  Confidence: {result['confidence']}%")
            print(f"  Priority  : {result['priority']}")
            print(f"  Needs review: {'YES' if result['needs_review'] else 'No'}")
    except EOFError:
        pass


if __name__ == "__main__":
    run_demo()
