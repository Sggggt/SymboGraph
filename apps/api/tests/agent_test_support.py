class Admission:
    """Minimal admitted-lease stub for current intent-execution tests."""

    def raise_if_lost(self):
        pass

    async def run(self, operation):
        return await operation

    async def release(self):
        pass
