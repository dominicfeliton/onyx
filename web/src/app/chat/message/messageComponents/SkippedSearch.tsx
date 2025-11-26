import SvgSearch from "@/icons/search";
import Button from "@/refresh-components/buttons/Button";
import Text from "@/refresh-components/texts/Text";

interface SkippedSearchProps {
  onForceSearch: () => void;
}

export default function SkippedSearch({ onForceSearch }: SkippedSearchProps) {
  return (
    <div className="flex items-center gap-2 p-3 bg-background-tint-01 rounded-lg border border-border-medium mb-3">
      <SvgSearch className="w-4 h-4 stroke-text-03 flex-shrink-0" />
      <Text text03 className="flex-1">
        The AI decided this query didn&apos;t need a search
      </Text>
      <Button secondary onClick={onForceSearch}>
        Force Search
      </Button>
    </div>
  );
}
