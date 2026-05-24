"""
PubMed document fetcher for BioASQ pipeline
Fetches abstracts and metadata from PubMed using Entrez API
"""

import time
import requests
from typing import List, Dict, Any, Optional
from xml.etree import ElementTree as ET


class PubMedFetcher:
    """Fetch PubMed documents using Entrez E-utilities API"""
    
    def __init__(
        self, 
        email: str = "user@example.com",
        api_key: Optional[str] = None,
        delay: float = 0.34
    ):
        """
        Initialize PubMed fetcher
        
        Args:
            email: Email for NCBI Entrez (required by NCBI)
            api_key: Optional NCBI API key for higher rate limits
            delay: Delay between requests (0.34s = 3 req/sec without API key)
        """
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.email = email
        self.api_key = api_key
        self.delay = delay
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.delay:
            time.sleep(self.delay - time_since_last)
        self.last_request_time = time.time()
    
    def fetch_abstracts(
        self, 
        pmids: List[str],
        batch_size: int = 200
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch abstracts for a list of PubMed IDs
        
        Args:
            pmids: List of PubMed IDs
            batch_size: Number of IDs to fetch per request
        
        Returns:
            Dictionary mapping PMID -> {title, abstract, pub_date, authors, journal}
        """
        results = {}
        
        # Process in batches
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            batch_results = self._fetch_batch(batch)
            results.update(batch_results)
        
        return results
    
    def _fetch_batch(self, pmids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch a batch of PubMed documents"""
        if not pmids:
            return {}
        
        self._rate_limit()
        
        # Build efetch URL
        ids = ",".join(pmids)
        params = {
            "db": "pubmed",
            "id": ids,
            "retmode": "xml",
            "rettype": "abstract",
            "email": self.email
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        url = f"{self.base_url}/efetch.fcgi"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            return self._parse_pubmed_xml(response.text)
        
        except Exception as e:
            print(f"Error fetching PubMed batch: {e}")
            return {}
    
    def _parse_pubmed_xml(self, xml_text: str) -> Dict[str, Dict[str, Any]]:
        """Parse PubMed XML response"""
        results = {}
        
        try:
            root = ET.fromstring(xml_text)
            
            for article in root.findall(".//PubmedArticle"):
                pmid_elem = article.find(".//PMID")
                if pmid_elem is None:
                    continue
                
                pmid = pmid_elem.text
                
                # Extract title
                title_elem = article.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None else ""
                
                # Extract abstract
                abstract_parts = []
                for abstract_text in article.findall(".//AbstractText"):
                    label = abstract_text.get("Label", "")
                    text = abstract_text.text or ""
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
                
                abstract = " ".join(abstract_parts)
                
                # Extract publication date
                pub_date = self._extract_pub_date(article)
                
                # Extract authors
                authors = self._extract_authors(article)
                
                # Extract journal
                journal_elem = article.find(".//Journal/Title")
                journal = journal_elem.text if journal_elem is not None else ""
                
                # Extract DOI
                doi = ""
                for article_id in article.findall(".//ArticleId"):
                    if article_id.get("IdType") == "doi":
                        doi = article_id.text
                        break
                
                results[pmid] = {
                    "doc_id": pmid,
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "pub_date": pub_date,
                    "authors": authors,
                    "journal": journal,
                    "doi": doi,
                    "source": "pubmed"
                }
        
        except Exception as e:
            print(f"Error parsing PubMed XML: {e}")
        
        return results
    
    def _extract_pub_date(self, article) -> str:
        """Extract publication date from article"""
        # Try ArticleDate first
        article_date = article.find(".//ArticleDate")
        if article_date is not None:
            year = article_date.find("Year")
            month = article_date.find("Month")
            day = article_date.find("Day")
            
            if year is not None:
                y = year.text
                m = month.text.zfill(2) if month is not None else "01"
                d = day.text.zfill(2) if day is not None else "01"
                return f"{y}-{m}-{d}"
        
        # Fall back to PubDate
        pub_date = article.find(".//PubDate")
        if pub_date is not None:
            year = pub_date.find("Year")
            month = pub_date.find("Month")
            
            if year is not None:
                y = year.text
                # Convert month name to number if needed
                m = self._month_to_number(month.text) if month is not None else "01"
                return f"{y}-{m}-01"
        
        return "2000-01-01"  # Default date
    
    def _month_to_number(self, month_str: str) -> str:
        """Convert month name to number"""
        months = {
            "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
            "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
            "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
        }
        return months.get(month_str[:3], month_str.zfill(2))
    
    def _extract_authors(self, article) -> List[str]:
        """Extract author names"""
        authors = []
        for author in article.findall(".//Author"):
            last_name = author.find("LastName")
            fore_name = author.find("ForeName")
            
            if last_name is not None:
                name = last_name.text
                if fore_name is not None:
                    name = f"{fore_name.text} {name}"
                authors.append(name)
        
        return authors
    
    def save_to_jsonl(
        self, 
        documents: Dict[str, Dict[str, Any]], 
        output_path: str
    ):
        """Save documents to JSONL format"""
        import json
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in documents.values():
                f.write(json.dumps(doc) + '\n')
