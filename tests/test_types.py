"""
Tests for PSE types and serialization.
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from engine.types import (
    FVG,
    MSS,
    Bar,
    FVGType,
    MSSDirection,
    SetupProposal,
    SetupType,
    Side,
    SwingPoint,
    SwingType,
    deserialize_from_json,
    serialize_to_json,
)


class TestTypes:
    """Test cases for type validation and serialization."""

    def test_bar_creation(self):
        """Test bar creation and validation."""
        bar = Bar(
            symbol="ES",
            timeframe="5m",
            session="RTH",
            venue="CME",
            timestamp=datetime.now(UTC),
            open=Decimal('100.00'),
            high=Decimal('101.00'),
            low=Decimal('99.00'),
            close=Decimal('100.50'),
            volume=1000
        )
        assert bar.high >= bar.low
        assert bar.high >= max(bar.open, bar.close, bar.low)
        assert bar.low <= min(bar.open, bar.close, bar.high)

    def test_bar_validation_errors(self):
        """Test bar validation errors."""
        with pytest.raises(ValueError):
            Bar(
                symbol="ES",
                timeframe="5m",
                session="RTH",
                venue="CME",
                timestamp=datetime.now(UTC),
                open=Decimal('100.00'),
                high=Decimal('99.00'),  # Invalid: high < low
                low=Decimal('100.00'),
                close=Decimal('100.50'),
                volume=1000
            )

    def test_swing_point_creation(self):
        """Test swing point creation."""
        swing = SwingPoint(
            bar_index=100,
            timestamp=datetime.now(UTC),
            price=Decimal('100.50'),
            swing_type=SwingType.SWING_HIGH,
            strength=8
        )
        assert swing.strength == 8
        assert swing.swing_type == SwingType.SWING_HIGH

    def test_mss_creation(self):
        """Test MSS creation."""
        mss = MSS(
            start_bar=50,
            end_bar=100,
            direction=MSSDirection.BULLISH,
            break_price=Decimal('100.00'),
            confirmation_price=Decimal('101.00'),
            is_valid=True
        )
        assert mss.start_bar <= mss.end_bar

    def test_fvg_creation(self):
        """Test FVG creation."""
        fvg = FVG(
            start_bar=75,
            end_bar=80,
            fvg_type=FVGType.BULLISH,
            top=Decimal('101.00'),
            bottom=Decimal('100.00'),
            is_filled=False
        )
        assert fvg.top > fvg.bottom

    def test_setup_proposal_creation(self):
        """Test setup proposal creation."""
        setup = SetupProposal(
            symbol="ES",
            setup_type=SetupType.BREAK_RETEST,
            side=Side.BUY,
            entry_price=Decimal('100.00'),
            stop_loss=Decimal('99.00'),
            take_profit=Decimal('102.00'),
            risk_reward_ratio=Decimal('2.0'),
            confidence=Decimal('0.85'),
            swing_points=[],
            mss_list=[],
            fvgs=[],
            volume_analysis={},
            order_flow={}
        )
        assert setup.risk_reward_ratio == Decimal('2.0')

    def test_setup_proposal_price_validation(self):
        """Test setup proposal price validation."""
        with pytest.raises(ValueError):
            SetupProposal(
                symbol="ES",
                setup_type=SetupType.BREAK_RETEST,
                side=Side.BUY,
                entry_price=Decimal('100.00'),
                stop_loss=Decimal('101.00'),  # Invalid: stop > entry for BUY
                take_profit=Decimal('102.00'),
                risk_reward_ratio=Decimal('2.0'),
                confidence=Decimal('0.85'),
                swing_points=[],
                mss_list=[],
                fvgs=[],
                volume_analysis={},
                order_flow={}
            )

    def test_deterministic_serialization(self):
        """Test deterministic JSON serialization."""
        bar1 = Bar(
            symbol="ES",
            timeframe="1m",
            session="RTH",
            venue="CME",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            open=Decimal('100.00'),
            high=Decimal('101.00'),
            low=Decimal('99.00'),
            close=Decimal('100.50'),
            volume=1000
        )

        bar2 = Bar(
            symbol="ES",
            timeframe="1m",
            session="RTH",
            venue="CME",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            open=Decimal('100.00'),
            high=Decimal('101.00'),
            low=Decimal('99.00'),
            close=Decimal('100.50'),
            volume=1000
        )

        json1 = serialize_to_json(bar1)
        json2 = serialize_to_json(bar2)
        assert json1 == json2

        restored = deserialize_from_json(json1, Bar)
        assert restored.timestamp == bar1.timestamp
        assert restored.open == bar1.open

    def test_json_roundtrip(self):
        """Test JSON roundtrip serialization."""
        setup = SetupProposal(
            symbol="NQ",
            setup_type=SetupType.FVG,
            side=Side.SELL,
            entry_price=Decimal('15000.00'),
            stop_loss=Decimal('15100.00'),
            take_profit=Decimal('14800.00'),
            risk_reward_ratio=Decimal('2.0'),
            confidence=Decimal('0.75'),
            swing_points=[],
            mss_list=[],
            fvgs=[],
            volume_analysis={'volume_spike': True},
            order_flow={'imbalance': 3.5}
        )

        json_str = serialize_to_json(setup)
        restored = deserialize_from_json(json_str, SetupProposal)

        assert restored.symbol == setup.symbol
        assert restored.side == setup.side
        assert restored.entry_price == setup.entry_price
        assert restored.volume_analysis == setup.volume_analysis


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
