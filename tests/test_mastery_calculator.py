import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MasteryCalculatorIntegrationTests(unittest.TestCase):
    def test_backend_files_are_valid_python(self):
        for path in (
            ROOT / "autopcr" / "util" / "mastery.py",
            ROOT / "autopcr" / "http_server" / "httpserver.py",
            ROOT / "server.py",
        ):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_page_and_api_are_registered(self):
        source = (ROOT / "autopcr" / "http_server" / "httpserver.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("'/account/<string:acc>/mastery'", source)
        self.assertIn("@self.web.route('/mastery')", source)
        self.assertTrue(
            (ROOT / "autopcr" / "http_server" / "ClientApp" / "mastery.html").is_file()
        )

    def test_qq_entry_and_read_only_notice_are_present(self):
        server = (ROOT / "server.py").read_text(encoding="utf-8")
        page = (
            ROOT / "autopcr" / "http_server" / "ClientApp" / "mastery.html"
        ).read_text(encoding="utf-8")
        self.assertIn('on_fullmatch(f"{prefix}专精计算器")', server)
        self.assertIn("不会向游戏提交强化操作", page)
        self.assertIn("3:1", page)
        self.assertIn("不足时使用万能碎片", page)


if __name__ == "__main__":
    unittest.main()
