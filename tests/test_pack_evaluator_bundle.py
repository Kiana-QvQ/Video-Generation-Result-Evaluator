from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import evaluator.core.paths as paths
import scripts.pack_evaluator_bundle as pack


class EvaluatorBundleTests(unittest.TestCase):
    def test_profile_provenance_paths_are_portable(self) -> None:
        project_file = str(
            pack.PROJECT_ROOT / "data" / "au" / "sample.csv"
        )
        external_file = r"C:\outside\sample.csv"

        portable = pack._portable_profile_value(
            {
                "project": project_file,
                "external": external_file,
                "nested": [project_file],
            }
        )

        self.assertEqual(portable["project"], "data/au/sample.csv")
        self.assertEqual(portable["external"], "sample.csv")
        self.assertEqual(portable["nested"], ["data/au/sample.csv"])

    def test_all_runtime_profile_assets_are_present(self) -> None:
        status = paths.verify_bundled_profiles()
        self.assertTrue(all(status.values()), status)
        self.assertTrue(
            paths.profile_path(
                "original_emotion_au_profile",
                required=True,
            ).is_file()
        )

    def test_public_entrypoint_runtime_file_is_present(self) -> None:
        for relative in pack.REQUIRED_PACKAGE_FILES:
            self.assertTrue(
                (pack.PACKAGE_ROOT / relative).is_file(),
                relative,
            )

    def test_zip_contains_only_verified_profiles_and_is_self_contained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            package_root = root / "package" / "evaluator"
            real_root = Path(__file__).resolve().parents[1]

            shutil.copytree(real_root / "evaluator", package_root)
            (package_root / "assets" / "profiles" / "stale.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            for filename, relative in pack.SOURCE_PROFILES.items():
                source = real_root / relative
                target = source_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

            output = root / "evaluator_bundle.zip"
            with (
                patch.object(pack, "PROJECT_ROOT", source_root),
                patch.object(pack, "PACKAGE_ROOT", package_root),
                patch.object(
                    pack,
                    "PROFILES_DIR",
                    package_root / "assets" / "profiles",
                ),
                patch.object(
                    paths,
                    "PROFILES_DIR",
                    package_root / "assets" / "profiles",
                ),
            ):
                archive = pack.build_bundle(output)

            self.assertEqual(archive.resolve(), output.resolve())
            self.assertTrue(archive.is_file())
            with zipfile.ZipFile(archive) as handle:
                names = set(handle.namelist())
                self.assertIn(
                    "evaluator/assets/profiles/original_emotion_au_profile.json",
                    names,
                )
                self.assertNotIn(
                    "evaluator/assets/profiles/stale.json",
                    names,
                )
                manifest = json.loads(
                    handle.read("evaluator/assets/MANIFEST.json")
                )
                source_profile = handle.read(
                    "evaluator/assets/profiles/wangxing_source_profile.json"
                )
            self.assertIn("original_emotion_au_profile", manifest["profiles"])
            self.assertEqual(
                set(manifest["profiles"]),
                set(paths.PROFILE_FILES),
            )
            self.assertNotIn(
                str(real_root).replace("\\", "\\\\").encode("utf-8"),
                source_profile,
            )

    def test_bundle_output_cannot_overwrite_evaluator_source(self) -> None:
        with self.assertRaises(SystemExit):
            pack._safe_staging_path(pack.PACKAGE_ROOT / "build")


if __name__ == "__main__":
    unittest.main()
