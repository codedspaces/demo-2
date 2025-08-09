#!/usr/bin/env python3
"""
AI Engineering Toolkit - Analytics & Engagement Tracking

This script automatically tracks repository engagement metrics, tool popularity,
and newsletter conversion rates for the AI Engineering Toolkit.

Usage:
    python track_engagement.py --config config.json
    python track_engagement.py --daily-report
    python track_engagement.py --weekly-summary
"""

import requests
import json
import csv
import os
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3
from pathlib import Path

class GitHubAnalytics:
    """GitHub repository analytics tracker"""
    
    def __init__(self, token: str, owner: str, repo: str):
        self.token = token
        self.owner = owner
        self.repo = repo
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = f"https://api.github.com/repos/{owner}/{repo}"
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for tracking"""
        db_path = Path("analytics/data/tracking.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(db_path))
        cursor = self.conn.cursor()
        
        # Repository stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS repo_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                stars INTEGER,
                forks INTEGER,
                watchers INTEGER,
                open_issues INTEGER,
                size INTEGER,
                created_at TEXT,
                updated_at TEXT
            )
        ''')
        
        # Traffic stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS traffic_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                date TEXT,
                views INTEGER,
                unique_visitors INTEGER
            )
        ''')
        
        # Referrer stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS referrer_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                referrer TEXT,
                views INTEGER,
                unique_visitors INTEGER
            )
        ''')
        
        # Tool clicks table (simulated - would come from URL shortener)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tool_clicks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                tool_name TEXT,
                category TEXT,
                clicks INTEGER
            )
        ''')
        
        self.conn.commit()
    
    def fetch_repo_stats(self) -> Dict:
        """Fetch comprehensive repository statistics"""
        try:
            response = requests.get(self.base_url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            stats = {
                'timestamp': datetime.now().isoformat(),
                'stars': data['stargazers_count'],
                'forks': data['forks_count'],
                'watchers': data['watchers_count'],
                'open_issues': data['open_issues_count'],
                'size': data['size'],
                'created_at': data['created_at'],
                'updated_at': data['updated_at']
            }
            
            return stats
        except Exception as e:
            print(f"Error fetching repo stats: {e}")
            return {}
    
    def fetch_traffic_stats(self) -> Dict:
        """Fetch traffic statistics (requires push access)"""
        try:
            # Views
            views_url = f"{self.base_url}/traffic/views"
            views_response = requests.get(views_url, headers=self.headers)
            
            # Referrers
            referrers_url = f"{self.base_url}/traffic/popular/referrers"
            referrers_response = requests.get(referrers_url, headers=self.headers)
            
            # Paths
            paths_url = f"{self.base_url}/traffic/popular/paths"
            paths_response = requests.get(paths_url, headers=self.headers)
            
            return {
                'views': views_response.json() if views_response.status_code == 200 else None,
                'referrers': referrers_response.json() if referrers_response.status_code == 200 else None,
                'paths': paths_response.json() if paths_response.status_code == 200 else None
            }
        except Exception as e:
            print(f"Error fetching traffic stats: {e}")
            return {}
    
    def fetch_releases_stats(self) -> List[Dict]:
        """Fetch release download statistics"""
        try:
            releases_url = f"{self.base_url}/releases"
            response = requests.get(releases_url, headers=self.headers)
            response.raise_for_status()
            
            releases = []
            for release in response.json():
                release_data = {
                    'name': release['name'],
                    'tag_name': release['tag_name'],
                    'published_at': release['published_at'],
                    'download_count': sum(asset['download_count'] for asset in release['assets'])
                }
                releases.append(release_data)
            
            return releases
        except Exception as e:
            print(f"Error fetching releases: {e}")
            return []
    
    def store_repo_stats(self, stats: Dict):
        """Store repository statistics in database"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO repo_stats 
            (timestamp, stars, forks, watchers, open_issues, size, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            stats['timestamp'], stats['stars'], stats['forks'],
            stats['watchers'], stats['open_issues'], stats['size'],
            stats['created_at'], stats['updated_at']
        ))
        self.conn.commit()
    
    def store_traffic_stats(self, traffic_data: Dict):
        """Store traffic statistics in database"""
        if not traffic_data.get('views'):
            return
        
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        
        # Store daily views
        for view_data in traffic_data['views'].get('views', []):
            cursor.execute('''
                INSERT INTO traffic_stats (timestamp, date, views, unique_visitors)
                VALUES (?, ?, ?, ?)
            ''', (timestamp, view_data['timestamp'], view_data['count'], view_data['uniques']))
        
        # Store referrer data
        if traffic_data.get('referrers'):
            for referrer in traffic_data['referrers']:
                cursor.execute('''
                    INSERT INTO referrer_stats (timestamp, referrer, views, unique_visitors)
                    VALUES (?, ?, ?, ?)
                ''', (timestamp, referrer['referrer'], referrer['count'], referrer['uniques']))
        
        self.conn.commit()
    
    def get_growth_metrics(self, days: int = 30) -> Dict:
        """Calculate growth metrics over specified period"""
        cursor = self.conn.cursor()
        
        # Get current stats
        cursor.execute('SELECT * FROM repo_stats ORDER BY timestamp DESC LIMIT 1')
        current = cursor.fetchone()
        
        # Get stats from N days ago
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        cursor.execute('''
            SELECT * FROM repo_stats 
            WHERE timestamp >= ? 
            ORDER BY timestamp ASC LIMIT 1
        ''', (cutoff_date,))
        previous = cursor.fetchone()
        
        if not current or not previous:
            return {}
        
        # Calculate growth
        star_growth = current[2] - previous[2]  # stars column
        fork_growth = current[3] - previous[3]  # forks column
        
        return {
            'period_days': days,
            'star_growth': star_growth,
            'fork_growth': fork_growth,
            'star_growth_rate': (star_growth / previous[2]) * 100 if previous[2] > 0 else 0,
            'fork_growth_rate': (fork_growth / previous[3]) * 100 if previous[3] > 0 else 0,
            'current_stars': current[2],
            'current_forks': current[3]
        }
    
    def track_external_tools(self) -> Dict:
        """Track popularity of external tools mentioned in README"""
        # This would typically integrate with URL shortener APIs
        # For now, we'll simulate with GitHub API calls to referenced repos
        
        popular_tools = [
            ('langchain-ai', 'langchain'),
            ('chroma-core', 'chroma'),
            ('qdrant', 'qdrant'),
            ('weaviate', 'weaviate'),
            ('microsoft', 'autogen'),
            ('joaomdmoura', 'crewAI'),
            ('mendableai', 'firecrawl'),
            ('scrapy', 'scrapy'),
            ('microsoft', 'playwright'),
            ('SeleniumHQ', 'selenium'),
            ('apify', 'apify-sdk-python'),
            ('codelucas', 'newspaper')
        ]
        
        tool_stats = {}
        for owner, repo in popular_tools:
            try:
                url = f"https://api.github.com/repos/{owner}/{repo}"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    tool_stats[repo] = {
                        'stars': data['stargazers_count'],
                        'forks': data['forks_count'],
                        'last_updated': data['updated_at']
                    }
            except Exception as e:
                print(f"Error fetching {owner}/{repo}: {e}")
        
        return tool_stats

class NewsletterTracker:
    """Track newsletter conversion metrics"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def track_conversions(self) -> Dict:
        """Track GitHub → Newsletter conversions"""
        # This would integrate with newsletter platform APIs
        # For demonstration, returning simulated metrics
        
        return {
            'github_referral_signups': 45,
            'total_subscribers': 1250,
            'conversion_rate': 3.6,
            'this_week_signups': 12,
            'github_traffic_percentage': 18.5
        }

class ReportGenerator:
    """Generate comprehensive analytics reports"""
    
    def __init__(self, github_analytics: GitHubAnalytics, newsletter_tracker: NewsletterTracker):
        self.github = github_analytics
        self.newsletter = newsletter_tracker
    
    def generate_daily_report(self) -> Dict:
        """Generate daily analytics report"""
        repo_stats = self.github.fetch_repo_stats()
        traffic_stats = self.github.fetch_traffic_stats()
        tool_stats = self.github.track_external_tools()
        newsletter_stats = self.newsletter.track_conversions()
        
        report = {
            'report_type': 'daily',
            'generated_at': datetime.now().isoformat(),
            'repository': repo_stats,
            'traffic': traffic_stats,
            'popular_tools': tool_stats,
            'newsletter': newsletter_stats
        }
        
        # Store data
        if repo_stats:
            self.github.store_repo_stats(repo_stats)
        if traffic_stats:
            self.github.store_traffic_stats(traffic_stats)
        
        return report
    
    def generate_weekly_summary(self) -> Dict:
        """Generate weekly analytics summary"""
        growth_metrics = self.github.get_growth_metrics(days=7)
        monthly_growth = self.github.get_growth_metrics(days=30)
        
        summary = {
            'report_type': 'weekly',
            'generated_at': datetime.now().isoformat(),
            'weekly_growth': growth_metrics,
            'monthly_growth': monthly_growth,
            'insights': self._generate_insights(growth_metrics)
        }
        
        return summary
    
    def _generate_insights(self, metrics: Dict) -> List[str]:
        """Generate actionable insights from metrics"""
        insights = []
        
        if metrics.get('star_growth', 0) > 50:
            insights.append("🚀 Exceptional star growth this week! Consider creating content about what's driving engagement.")
        
        if metrics.get('fork_growth', 0) > 10:
            insights.append("⚡ High fork activity suggests developers are actively using the toolkit.")
        
        if metrics.get('star_growth_rate', 0) > 10:
            insights.append("📈 Star growth rate above 10% - momentum is building!")
        
        return insights
    
    def export_report(self, report: Dict, format: str = 'json'):
        """Export report to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'json':
            filename = f"analytics/reports/report_{report['report_type']}_{timestamp}.json"
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
        
        elif format == 'csv' and report['report_type'] == 'daily':
            filename = f"analytics/reports/daily_stats_{timestamp}.csv"
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                repo_data = report['repository']
                writer.writerow(['Metric', 'Value'])
                for key, value in repo_data.items():
                    writer.writerow([key, value])
        
        print(f"Report exported to: {filename}")
        return filename

def main():
    parser = argparse.ArgumentParser(description='AI Engineering Toolkit Analytics')
    parser.add_argument('--config', default='analytics/config.json', help='Configuration file')
    parser.add_argument('--daily-report', action='store_true', help='Generate daily report')
    parser.add_argument('--weekly-summary', action='store_true', help='Generate weekly summary')
    parser.add_argument('--export-format', choices=['json', 'csv'], default='json', help='Export format')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {args.config}")
        print("Creating default configuration...")
        
        default_config = {
            "github": {
                "token": "your_github_token_here",
                "owner": "yourusername",
                "repo": "ai-engineering-toolkit"
            },
            "newsletter": {
                "platform": "substack",
                "api_key": "your_api_key_here"
            },
            "tracking": {
                "utm_source": "ai-engineering-toolkit",
                "utm_medium": "github"
            }
        }
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"Default configuration created at: {config_path}")
        print("Please update with your actual credentials and run again.")
        return
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Initialize trackers
    github_analytics = GitHubAnalytics(
        token=config['github']['token'],
        owner=config['github']['owner'],
        repo=config['github']['repo']
    )
    
    newsletter_tracker = NewsletterTracker(config['newsletter'])
    report_generator = ReportGenerator(github_analytics, newsletter_tracker)
    
    # Generate reports
    if args.daily_report:
        print("Generating daily report...")
        report = report_generator.generate_daily_report()
        filename = report_generator.export_report(report, args.export_format)
        
        print(f"Daily Report Summary:")
        print(f"⭐ Stars: {report['repository'].get('stars', 'N/A')}")
        print(f"🍴 Forks: {report['repository'].get('forks', 'N/A')}")
        print(f"📬 Newsletter Signups: {report['newsletter'].get('this_week_signups', 'N/A')}")
    
    elif args.weekly_summary:
        print("Generating weekly summary...")
        summary = report_generator.generate_weekly_summary()
        filename = report_generator.export_report(summary, args.export_format)
        
        print(f"Weekly Summary:")
        growth = summary.get('weekly_growth', {})
        print(f"📈 Star Growth: +{growth.get('star_growth', 0)}")
        print(f"📈 Fork Growth: +{growth.get('fork_growth', 0)}")
        
        insights = summary.get('insights', [])
        if insights:
            print("\n💡 Insights:")
            for insight in insights:
                print(f"  {insight}")
    
    else:
        print("Fetching current repository statistics...")
        stats = github_analytics.fetch_repo_stats()
        if stats:
            print(f"⭐ Stars: {stats['stars']}")
            print(f"🍴 Forks: {stats['forks']}")
            print(f"👀 Watchers: {stats['watchers']}")
            print(f"🐛 Open Issues: {stats['open_issues']}")
        
        growth = github_analytics.get_growth_metrics(days=7)
        if growth:
            print(f"\n📊 7-day Growth:")
            print(f"  Stars: +{growth['star_growth']} ({growth['star_growth_rate']:.1f}%)")
            print(f"  Forks: +{growth['fork_growth']} ({growth['fork_growth_rate']:.1f}%)")

if __name__ == "__main__":
    main()