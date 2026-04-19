export default function SectionTitle({ title }: { title: string }) {
  return (
    <h2 className="text-2xl lg:text-4xl bg-base-200/80 py-4 text-center font-bold mb-8">
      ⭐ {title} ⭐
      <span className="text-sm text-base-content/70 italic hidden lg:block">
        Shri Someshwar Shikshan Prasarak Mandal's Sharadchandra Pawar College of Engineering & Technology, Delhi
      </span>
    </h2>
  );
}
