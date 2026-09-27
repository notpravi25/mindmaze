"""
Unit tests for the Text Normalization Engine.
Verifies all legal suffix rules, punctuation normalization, domain handling,
social handles, word-order invariance, Unicode/multilingual safety, and address extraction.
"""

import sys
import os
import unittest

# Ensure parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.normalize import (
    normalize_unicode,
    clean_text,
    strip_legal_suffixes,
    normalize_domain_or_handle,
    to_alphanumeric,
    to_sorted_tokens,
    tokenize,
    extract_numbers,
    extract_postal_code,
    normalize_business_name,
    normalize_address,
    normalize_country
)


class TestTextNormalizationEngine(unittest.TestCase):

    def test_legal_suffixes_multilingual(self):
        """Test removal of legal suffixes across US, UK, India, and France."""
        cases = [
            ("ABC Inc.", "abc"),
            ("ABC LLC", "abc"),
            ("ABC L.L.C.", "abc"),
            ("ABC Corp", "abc"),
            ("ABC Corporation", "abc"),
            ("ABC Pvt Ltd", "abc"),
            ("ABC Pvt. Ltd.", "abc"),
            ("ABC Private Limited", "abc"),
            ("ABC SARL", "abc"),
            ("ABC S.A.R.L.", "abc"),
            ("ABC SAS", "abc"),
            ("ABC SASU", "abc"),
            ("ABC SCI", "abc"),
            ("ABC Technologies Ltd", "abc"),
            ("ABC Solutions Pvt Ltd", "abc"),
            ("ABC Enterprises Inc", "abc")
        ]
        for raw, expected in cases:
            cleaned = strip_legal_suffixes(raw)
            self.assertEqual(cleaned, expected, f"Failed for {raw}: got '{cleaned}', expected '{expected}'")

    def test_suffix_not_removed_from_middle_of_words(self):
        """Ensure boundary-awareness: words like 'salvage', 'incorporate' are not damaged."""
        self.assertEqual(strip_legal_suffixes("Salvage Yard"), "salvage yard")
        self.assertEqual(strip_legal_suffixes("Incorporate Ideas"), "incorporate ideas")
        self.assertEqual(strip_legal_suffixes("South Africa Logistics"), "south africa logistics")

    def test_punctuation_and_noise_brackets(self):
        """Test normalization of punctuation, brackets [[LLC]], ##, and <<."""
        self.assertEqual(clean_text("ABC-Pvt-Ltd"), "abc pvt ltd")
        self.assertEqual(clean_text("ABC, Pvt Ltd"), "abc pvt ltd")
        self.assertEqual(clean_text("ABC. Pvt. Ltd."), "abc pvt ltd")
        self.assertEqual(clean_text("AMERICAN CHOICE SERVICE [[LLC]]"), "american choice service llc")
        self.assertEqual(clean_text("##2476 Main St"), "2476 main st")
        self.assertEqual(clean_text("<< Team Ecole"), "team ecole")
        self.assertEqual(clean_text("Barnes & Noble"), "barnes and noble")

    def test_domain_normalization(self):
        """Test extraction of business core from domain names."""
        self.assertEqual(normalize_domain_or_handle("strategicpraetorian.com"), "strategicpraetorian")
        self.assertEqual(normalize_domain_or_handle("company.co.in"), "company")
        self.assertEqual(normalize_domain_or_handle("organization.org"), "organization")
        self.assertEqual(normalize_domain_or_handle("https://www.testbiz.in"), "testbiz")

    def test_social_handles(self):
        """Test handling of social media usernames / handles."""
        self.assertEqual(normalize_domain_or_handle("@smartraj"), "smartraj")
        self.assertEqual(normalize_domain_or_handle("@apex_labs"), "apex labs")

    def test_token_sorting_order_invariance(self):
        """Test word-order invariance (transposed words yield identical sorted tokens)."""
        s1 = to_sorted_tokens("XX Apex Nippon", remove_suffixes=True)
        s2 = to_sorted_tokens("XX Nippon Apex", remove_suffixes=True)
        self.assertEqual(s1, s2)
        self.assertEqual(s1, "apex nippon xx")

    def test_unicode_accents_and_indic_preservation(self):
        """Test that French accents fold while Indic characters and matras are preserved."""
        # French: accents should fold cleanly to base Latin
        self.assertEqual(normalize_unicode("Président"), "President")
        self.assertEqual(normalize_unicode("École"), "Ecole")
        self.assertEqual(normalize_unicode("Hêtres"), "Hetres")
        self.assertEqual(clean_text("Président École Hêtres"), "president ecole hetres")

        # Indic: Devanagari and Tamil must remain completely intact
        hindi = "शांति मॉडर्न फाइनेंस"
        self.assertEqual(normalize_unicode(hindi), hindi)
        self.assertEqual(clean_text(hindi), hindi)

        tamil = "யுனிவர்சல்"
        self.assertEqual(normalize_unicode(tamil), tamil)
        self.assertEqual(clean_text(tamil), tamil)

    def test_address_number_extraction(self):
        """Test extraction of building, plot, flat, and street numbers."""
        addr1 = "No. 35, Brentwood Apartments, 2Nd Main, Indiranagar, Bangalore"
        nums1 = extract_numbers(addr1)
        self.assertIn("35", nums1)
        self.assertIn("2", nums1)

        addr2 = "Plot No.51, No.9, Aster Court, Vgp Golden Beach, Chennai"
        nums2 = extract_numbers(addr2)
        self.assertIn("51", nums2)
        self.assertIn("9", nums2)

        addr3 = "702 1/2 N St, Washington, DC 20001"
        nums3 = extract_numbers(addr3)
        self.assertIn("702", nums3)
        self.assertIn("1/2", nums3)
        self.assertIn("20001", nums3)

    def test_postal_code_extraction(self):
        """Test extraction of 5-digit US ZIP, 6-digit India PIN, and 5-digit France postal codes."""
        # US ZIP
        self.assertEqual(extract_postal_code("702 N Street, Washington, DC 20001"), "20001")
        # India PIN
        self.assertEqual(extract_postal_code("MG Road, Bangalore, Karnataka 560001"), "560001")
        # France Postal Code
        self.assertEqual(extract_postal_code("63 R. DE DIEPPE, LILLE, 59000"), "59000")
        # No postal code
        self.assertIsNone(extract_postal_code("Cotten Road, Tyler, TX"))

    def test_address_abbreviations(self):
        """Test standardization of common street abbreviations."""
        addr_res = normalize_address("27 River St, Bingham, ME")
        self.assertIn("street", addr_res["address_tokens"])

        addr_res2 = normalize_address("63 R. DE DIEPPE, LILLE")
        self.assertIn("rue", addr_res2["address_tokens"])

    def test_country_normalization(self):
        """Test generic country normalization for both known and unseen countries."""
        self.assertEqual(normalize_country("US")["country_normalized"], "US")
        self.assertEqual(normalize_country("  india  ")["country_normalized"], "INDIA")
        self.assertEqual(normalize_country("France")["country_normalized"], "FRANCE")
        self.assertEqual(normalize_country("Germany")["country_normalized"], "GERMANY")

    def test_deterministic_normalization(self):
        """Ensure repeated normalization produces identical outputs."""
        sample = {
            "entity_id": "S1-925783039",
            "business_name": "Orelee's Barbershop & Salon Inc.",
            "business_address": "1795 Westchester Drive, High Point, NC 27262",
            "country": "US"
        }
        res1 = normalize_business_name(sample["business_name"])
        res2 = normalize_business_name(sample["business_name"])
        self.assertEqual(res1, res2)


if __name__ == "__main__":
    unittest.main()
