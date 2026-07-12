import unittest

from app.services.tools.sql_tool import _format_structured_answer


class SQLToolFormattingTests(unittest.TestCase):
    def test_customer_rows_include_business_identifiers(self) -> None:
        answer = _format_structured_answer(
            [
                {
                    "customer_id": 4,
                    "company_name": "Delta Logistics",
                    "subscription_tier": "Enterprise",
                    "account_status": "Suspended",
                    "sla_level": "Priority",
                    "region": "MEA",
                }
            ],
            "SELECT customer_id, company_name, subscription_tier, account_status, sla_level, region FROM customers",
        )

        self.assertIn("Delta Logistics", answer)
        self.assertIn("account_status=Suspended", answer)
        self.assertNotEqual("Found 1 matching structured record from customers.", answer)

    def test_incident_rows_include_operational_context(self) -> None:
        answer = _format_structured_answer(
            [
                {
                    "incident_id": 1,
                    "incident_type": "Outage",
                    "severity": "Critical",
                    "affected_region": "EU",
                    "resolution_status": "Investigating",
                    "root_cause": "Database cluster overload",
                }
            ],
            "SELECT incident_id, incident_type, severity, affected_region, resolution_status, root_cause FROM incident_logs",
        )

        self.assertIn("incident_type=Outage", answer)
        self.assertIn("affected_region=EU", answer)
        self.assertIn("root_cause=Database cluster overload", answer)

    def test_count_tuple_result_is_readable(self) -> None:
        answer = _format_structured_answer(
            [(3,)],
            "SELECT COUNT(*) FROM support_tickets WHERE ticket_status = 'Open'",
        )

        self.assertEqual("Count result from support_tickets: 3.", answer)

    def test_unknown_table_does_not_display_sensitive_columns(self) -> None:
        answer = _format_structured_answer(
            [{"token": "secret-token-value", "status": "Active"}],
            "SELECT token, status FROM unknown_table",
        )

        self.assertIn("status=Active", answer)
        self.assertNotIn("secret-token-value", answer)


if __name__ == "__main__":
    unittest.main()
